# # Parametric design study with optiSLang ProxySolver
#
# This example shows how to combine PyAEDT and pyoptiSLang to run a parametric
# sensitivity study on a dipole antenna in HFSS. optiSLang's ProxySolver node
# orchestrates the parallel design evaluations and trains a surrogate model (MOP)
# on the collected results.
#
# Keywords: **AEDT**, **HFSS**, **optiSLang**, **parametric**, **ProxySolver**, **sensitivity**

# ## Perform imports and define constants
#
# Import the required packages.

# +
import os
import pathlib
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map as concurrent_map

from ansys.aedt.core import Hfss
from ansys.optislang.core import Optislang
import ansys.optislang.core.node_types as node_types
from ansys.optislang.core.nodes import DesignFlow
from ansys.optislang.core.project_parametric import OptimizationParameter



# -

# ## Define constants.
#
# **Take note:** Jupyters execution model does not work well with multiprocessing.
# When executing from within Jupyter, please set `MAX_PARALLEL_SOLVE_PROCESSES = 1`, 
# otherwise the process will fail.
# When executed outside of Jupyter, `MAX_PARALLEL_SOLVE_PROCESSES` can be increased,
# as long as `NUM_CORES_PER_JOB * MAX_PARALLEL_SOLVE_PROCESSES` does not exceed the
# number of available cores.

AEDT_VERSION = "2026.1"
NUM_CORES_PER_PROCESS = 8
NG_MODE = True #True  # HFSS jobs run non-graphically (headless) to support parallel execution.
MAX_PARALLEL_SOLVE_PROCESSES = 1
AEDT_WORKING_DIRNAME = "pyaedt_workingdir"
SOLVE_MODE = "DUMMY"  # Set to "DUMMY" to run without an HFSS license, for testing purposes.

# ## Define HFSS solver function
#
# The ``solve_hfss()`` function creates and solves a dipole antenna model in HFSS
# for a given set of design parameters, exports the return loss to a CSV file,
# and returns the result as a NumPy array.
# Each call runs in its own HFSS desktop instance and releases it when done.
#
# > **Note:** This function must remain a module-level definition so that
# > ``multiprocessing.Pool`` can pickle it for parallel dispatch.


def solve_hfss(working_dir, l_dipole, wire_rad, port_gap):
    project_name = os.path.join(working_dir, "dipole.aedt")
    hfss = Hfss(
        version=AEDT_VERSION,
        non_graphical=NG_MODE,
        project=project_name,
        new_desktop=True,
        solution_type="Modal",
    )

    hfss["l_dipole"] = f"{l_dipole}cm"
    hfss["wire_rad"] = f"{wire_rad}mm"
    hfss["port_gap"] = f"{port_gap}mm"

    component_name = "Dipole_Antenna_DM"
    freq_range = ["1GHz", "2GHz"]
    center_freq = "1.5GHz"
    freq_step = "0.5GHz"

    component_fn = hfss.components3d[component_name]
    comp_params = hfss.get_component_variables(component_name)
    comp_params["dipole_length"] = "l_dipole"
    comp_params["wire_rad"] = "wire_rad"
    comp_params["port_gap"] = "port_gap"
    hfss.modeler.insert_3d_component(component_fn, geometry_parameters=comp_params)

    hfss.create_open_region(frequency=center_freq)

    setup_name = "MySetup"
    setup = hfss.create_setup(
        name=setup_name,
        MultipleAdaptiveFreqsSetup=freq_range,
        MaximumPasses=2,
    )
    setup.add_sweep(
        name="DiscreteSweep",
        sweep_type="Discrete",
        RangeStart=freq_range[0],
        RangeEnd=freq_range[1],
        RangeStep=freq_step,
        SaveFields=True,
    )
    interp_sweep = setup.add_sweep(
        name="InterpolatingSweep",
        sweep_type="Interpolating",
        RangeStart=freq_range[0],
        RangeEnd=freq_range[1],
        SaveFields=False,
    )

    hfss.analyze_setup(setup_name, use_auto_settings=True, cores=NUM_CORES_PER_PROCESS)
    hfss.create_scattering(plot="Return Loss", sweep=interp_sweep.name)
    hfss.post.export_report_to_file(working_dir, "Return Loss", ".csv")

    hfss.save_project()
    hfss.release_desktop(close_projects=True, close_desktop=True)

    data = np.loadtxt(os.path.join(working_dir, "Return Loss.csv"), delimiter=",", skiprows=1)
    return data

# ## Define solver wrappers
#
# ``call_solver()`` wraps ``solve_hfss()`` into the argument tuple format expected
# by ``multiprocessing.Pool.imap()``.
# ``call_solver_dummy()`` returns synthetic data and can be used to verify the
# workflow without an HFSS license (set ``SOLVE_MODE = "DUMMY"``).


def call_solver(args):
    hid, working_dir, l_dipole, wire_rad, port_gap = args
    print(f"Solving design {hid} ...")
    result_data = solve_hfss(working_dir, l_dipole, wire_rad, port_gap)
    freq = result_data[:, -2].tolist()
    loss = result_data[:, -1].tolist()
    return_loss = {
        "abscissa": freq,
        "channels": [loss],
        "num_channels": 1,
        "num_entries": len(freq),
        "type": "signal",
    }
    print(f"Solving design {hid} ... done.")
    return return_loss


def call_solver_dummy(args):
    hid, working_dir, l_dipole, wire_rad, port_gap = args
    print(f"Solving design {hid} ...")
    return_loss = {
        "abscissa": [1.0, 2.0, 3.0],
        "channels": [[i * l_dipole - wire_rad for i in [1.0, 2.0, 3.0]]],
        "num_channels": 1,
        "num_entries": 3,
        "type": "signal",
    }
    print(f"Solving design {hid} ... done.")
    return return_loss


# ## Define parallel compute function
#
# ``compute_designs()`` receives a list of design points from the ProxySolver,
# dispatches them to the solver in parallel using ``multiprocessing.Pool``,
# and returns the collected responses.


def get_parameter_value(parameter_list, parameter_name):
    for parameter in parameter_list:
        if parameter["name"] == parameter_name:
            return parameter["value"]


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
        
    except Exception:
        return False


def compute_designs(designs):
    print(f"Calculate {len(designs)} designs: {', '.join([design['hid'] for design in designs])}")
    design_data = []
    aedt_working_dir = WORKING_DIR / AEDT_WORKING_DIRNAME
    aedt_working_dir.mkdir(parents=True, exist_ok=True)
    for design in designs:
        hid = design["hid"]
        parameters = design["parameters"]
        design_temp_folder = tempfile.mkdtemp(dir=str(aedt_working_dir), prefix="pyaedt.")
        design_data.append((
            hid,
            design_temp_folder,
            get_parameter_value(parameters, "l_dipole"),
            get_parameter_value(parameters, "wire_rad"),
            get_parameter_value(parameters, "port_gap"),
        ))

    if SOLVE_MODE == "HFSS":
        solve = call_solver
    elif SOLVE_MODE == "DUMMY":
        solve = call_solver_dummy
    else:
        raise KeyError(f"Unknown SOLVE_MODE: {SOLVE_MODE}")

    if in_notebook():
        # Run sequentially in Jupyter to avoid multiprocessing issues.
        print(
            "Running in notebook environment -> Ignoring MAX_PARALLEL_SOLVE_PROCESSES "
            "and running processes in sequence"
        )
        results = []
        for design_args in tqdm(design_data, desc="Solving designs"):
            result = solve(design_args)
            results.append(result)
    else:
        results = concurrent_map(solve, design_data, max_workers=MAX_PARALLEL_SOLVE_PROCESSES)

    result_design_list = []
    for design, result in zip(design_data, results):
        result_design_list.append({
            "hid": design[0],
            "responses": [{"name": "return_loss", "value": result}],
        })

    print(f"Return {len(result_design_list)} designs")
    return result_design_list


# ## Define optiSLang helper functions
#
# The following functions build the optiSLang workflow graph:
#
# - ``add_sensitivity_system()`` adds a Sensitivity system.


def add_sensitivity_system(system_name, parent_system, num_designs_max=100):
    system = parent_system.create_node(type_=node_types.Sensitivity, name=system_name)
    settings = system.get_property("AlgorithmSettings")
    settings["num_discretization"] = num_designs_max
    system.set_property("AlgorithmSettings", settings)
    return system


# - ``add_oco_system()`` adds an **One-Click-Optimization (OCO)** system.


def add_oco_system(system_name, parent_system, num_designs_max=100, use_mop=False):
    system = parent_system.create_node(type_=node_types.OCO, name=system_name)
    settings = system.get_property("Settings")
    for item in settings["sequence"]:
        if item["First"] == "Maximum number of samples":
            item["Second"] = num_designs_max

        if item["First"] == "Use MOP":
            item["Second"] = use_mop
    system.set_property("Settings", settings)

    return system

# - ``add_proxysolver_node()`` adds and configures a ProxySolver node.


def add_proxysolver_node(parent_system):
    proxy_solver = parent_system.create_node(
        type_=node_types.ProxySolver,
        name="MyProxySolver",
        design_flow=DesignFlow.RECEIVE_SEND,
    )
    proxy_solver.set_property("MultiDesignLaunchNum", -1)  # -1 submits all designs simultaneously.
    proxy_solver.set_property("ForwardHPCLicenseContextEnvironment", True)
    return proxy_solver


# - ``add_mopsolver_node()`` adds and configures a MOPSolver node.


def add_mopsolver_node(parent_system):
    mop_solver = parent_system.create_node(
        type_=node_types.Mopsolver,
        name="MyMOPSolver",
        design_flow=DesignFlow.RECEIVE_SEND,
    )
    return mop_solver


# - ``add_mop_node()`` appends a metamodel-of-optimal-prognosis (MOP) node and
#   connects it to the upstream sensitivity system.


def add_mop_node(parent_system, predecessor_system):
    mop = parent_system.create_node(type_=node_types.Mop, name="MOP")
    predecessor_system.get_output_slots("OMDBPath")[0].connect_to(
        mop.get_input_slots("IMDBPath")[0]
    )
    predecessor_system.get_output_slots("OParameterManager")[0].connect_to(
        mop.get_input_slots("IParameterManager")[0]
    )
    return mop

# - ``prepare_parameter_response_schema()`` builds the JSON schema required to
#   initialize the ProxySolver with the study's parameter and response definitions.


def prepare_parameter_response_schema(parameters, responses):
    load_json = {"parameters": [], "responses": []}
    for parameter_name, parameter_data in parameters.items():
        load_json["parameters"].append({
            "dir": {"value": "input"},
            "name": parameter_name,
            "value": parameter_data["reference_value"],
        })
    for response_name in responses:
        load_json["responses"].append({"dir": {"value": "output"}, "name": response_name})
    return load_json


# ## Prepare run
#
# * Create a temporary directory where downloaded data or dumped data can be stored.
# * Define parameter and response definition
# * Define total number of designs to execute `num_designs_max`

if __name__ == "__main__":
    # Create temporary working dir
    temp_folder = tempfile.TemporaryDirectory(suffix=".ansys")
    WORKING_DIR = pathlib.Path(temp_folder.name)
    print(f"WORKING DIR: {WORKING_DIR}")

    # The optiSLang project file (``pyoptislang_example_proxy_solver.opf``) is created 
    # on the fly by optiSLang when the ``Optislang`` instance is initialized.
    osl_project_name = "pyoptislang_example_proxy_solver.opf"

    # The dipole antenna geometry is parameterized by three variables.
    # ``num_designs_max`` controls how many design points the sensitivity study evaluates.
    parameters = {
        "l_dipole": {"reference_value": 10.2, "lower_bound": 9.0, "upper_bound": 12.0},
        "wire_rad": {"reference_value": 1.0, "lower_bound": 0.8, "upper_bound": 1.2},
        "port_gap": {"reference_value": 1.0, "lower_bound": 0.8, "upper_bound": 1.2},
    }
    responses = {"return_loss": {}}
    num_designs_max = 20

# ## Initialize the optiSLang session
# Create optiSLang project named `osl_project_name` and create
#   * Sensitivity system with ProxySolver node
#   * MOP node
# Load the parameter/response schema.

if __name__ == "__main__":
    osl = Optislang(
        project_path=str(WORKING_DIR / osl_project_name),
        batch=NG_MODE,
    )
    root_system = osl.application.project.root_system
    
    system = add_sensitivity_system("Sensitivity", root_system, num_designs_max=num_designs_max)
    proxy_solver = add_proxysolver_node(system)
    
    proxy_solver.load(args=prepare_parameter_response_schema(parameters, responses))
    proxy_solver.register_locations_as_parameter()
    proxy_solver.register_locations_as_response()
    
    for parameter_name, parameter_data in parameters.items():
        system.parameter_manager.modify_parameter(
            OptimizationParameter(
                name=parameter_name,
                reference_value=parameter_data["reference_value"],
                range=(parameter_data["lower_bound"], parameter_data["upper_bound"]),
            )
        )
    
    osl.application.save()


# ## Run the parametric study
# Start the optiSLang workflow without blocking. Poll the ProxySolver for pending
# design batches, dispatch them to HFSS via ``compute_designs()``, and return the
# responses until all design points have been evaluated.

if __name__ == "__main__":
    osl.application.project.start(wait_for_finished=False)

    all_responses = []
    while True:
        print("Waiting for optiSLang to provide next batch of designs ..")
        time.sleep(1.0)
        design_list = proxy_solver.get_designs()
        responses_list = compute_designs(design_list)
        all_responses += responses_list
        proxy_solver.set_designs(responses_list)
        if len(all_responses) >= num_designs_max:
            break
    
    print("Parametric design study done!")


# ## Add surrogate model (MOP)
# After all designs have been evaluated, append a **Metamodel of optimal prognosis
# (MOP)** node to the workflow and re-start the project to train the surrogate model
# on the collected results.

if __name__ == "__main__":
    moptraining_node = add_mop_node(root_system, system)
    print("Added and connected MOP node.")
    
    osl.application.project.start()
    print("Surrogate model training complete.")


# ## Release optiSLang

# +

if __name__ == "__main__":
    osl.dispose()
    time.sleep(3)  # Allow optiSLang to shut down before cleaning the temporary project folder.

# -

# ## Display results

# +
if __name__ == "__main__":
    
    for response in all_responses:
        freq = response["responses"][0]["value"]["abscissa"]
        return_loss = response["responses"][0]["value"]["channels"][0]
        plt.plot(freq, return_loss, label=f"Design {response['hid']}")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Return loss [dB]")
    plt.grid(True)
    plt.legend()
    plt.show()


# -

# ## Clean up
# All project files are saved in the folder ``temp_folder.name``.
# If you've run this example as a Jupyter notebook, you can retrieve those
# project files. The following command all temporary files, including the project folder.

if __name__ == "__main__":
    pass
    #temp_folder.cleanup()
