import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import shutil
import tempfile
import time

# Modules so we can to take care of parallel computing ourselves
from multiprocessing import Pool
from tqdm import tqdm

# pyaedt
from ansys.aedt.core import Hfss

# pyoptislang
from ansys.optislang.core import Optislang
import ansys.optislang.core.node_types as node_types
from ansys.optislang.core.nodes import DesignFlow, ParametricSystem, ProxySolverNode
from ansys.optislang.core.project_parametric import (
    ComparisonType,
    ObjectiveCriterion,
    OptimizationParameter,
)
from ansys.optislang.core import Optislang


WORKING_DIR = pathlib.Path(os.getcwd())
MAX_PARALLEL_SOLVE_PROCESSES = 2
AEDT_WORKING_DIRNAME = "pyaedt_workingdir"

AEDT_VERSION = "2026.1"
NUM_CORES_PER_JOB = 4
NG_MODE = True  # NO AEDT UI
SOLVE_MODE = "DUMMY"


def solve_hfss(working_dir, l_dipole, wire_rad, port_gap):
    project_name = os.path.join(working_dir, "dipole.aedt")
    hfss = Hfss(
        version=AEDT_VERSION,
        non_graphical=NG_MODE,
        num_cores=NUM_CORES_PER_JOB,
        project=project_name,
        new_desktop=True,
        solution_type="Modal",
    )

    hfss["l_dipole"] = f"{l_dipole}cm"
    hfss["wire_rad"] = f"{wire_rad}mm"
    hfss["port_gap"] = f"{port_gap}mm"
    component_name = "Dipole_Antenna_DM"
    freq_range = ["1GHz", "2GHz"]  # Frequency range for analysis and post-processing.
    center_freq = "1.5GHz"  # Center frequency
    freq_step = "0.5GHz"

    component_fn = hfss.components3d[component_name]  # Full file name.
    comp_params = hfss.get_component_variables(component_name)  # Retrieve dipole parameters.
    comp_params["dipole_length"] = "l_dipole"  # Update the dipole length.
    comp_params["wire_rad"] = "wire_rad"  # Update the dipole length.
    comp_params["port_gap"] = "port_gap"  # Update the dipole length.
    hfss.modeler.insert_3d_component(component_fn, geometry_parameters=comp_params)

    hfss.create_open_region(frequency=center_freq)

    setup = hfss.create_setup(name="MySetup", MultipleAdaptiveFreqsSetup=freq_range, MaximumPasses=2)

    disc_sweep = setup.add_sweep(name="DiscreteSweep", sweep_type="Discrete", RangeStart=freq_range[0], RangeEnd=freq_range[1], RangeStep=freq_step, SaveFields=True)

    interp_sweep = setup.add_sweep(name="InterpolatingSweep", sweep_type="Interpolating", RangeStart=freq_range[0], RangeEnd=freq_range[1], SaveFields=False)

    setup.analyze()
    spar_plot = hfss.create_scattering(plot="Return Loss", sweep=interp_sweep.name)
    hfss.post.export_report_to_file(working_dir, "Return Loss", ".csv")

    hfss.save_project()
    hfss.release_desktop(close_projects=True, close_desktop=True)

    data = np.loadtxt(os.path.join(working_dir, "Return Loss.csv"), delimiter=",", skiprows=1)
    return data


def call_solver_dummy(args):
    hid, working_dir, l_dipole, wire_rad, port_gap = args
    print(f"Solving design {hid} ...")
    return_loss = {
            "abscissa" : [1.0,2.0,3.0],
            "channels" :
            [
                [i*l_dipole-wire_rad for i in [1.0,2.0,3.0]]
            ],
            "num_channels" : 1,
            "num_entries" : 3,
            "type" : "signal"
        }
    print(f"Solving design {hid} ... done.")
    return return_loss


def call_solver(args):
    hid, working_dir, l_dipole, wire_rad, port_gap = args
    result_data = solve_hfss(working_dir, l_dipole, wire_rad, port_gap)

    freq = result_data[:,-2].tolist()
    loss = result_data[:,-1].tolist()
    return_loss = {
            "abscissa" : freq,
            "channels" :
            [
                loss
            ],
            "num_channels" : 1,
            "num_entries" : len(freq),
            "type" : "signal"
        }
    return_loss_min = min(loss)
    print(f"Solving design {hid} ... done.")
    return return_loss


def get_parameter_value(parameter_list, parameter_name):
    for parameter in parameter_list:
        if parameter["name"] == parameter_name:
            return parameter["value"]


def compute_designs(designs):
    result_design_list = []
    print(f"Calculate {len(designs)} designs")
    design_data = []
    aedt_working_dir = WORKING_DIR / AEDT_WORKING_DIRNAME
    aedt_working_dir.mkdir(parents=True, exist_ok=True)
    for design in designs:
        hid = design["hid"]
        parameters = design["parameters"]

        temp_folder = tempfile.mkdtemp(dir=str(aedt_working_dir), prefix="pyaedt.")
        this_design_data = (
            hid,
            temp_folder, 
            get_parameter_value(parameters, "l_dipole"),
            get_parameter_value(parameters, "wire_rad"),
            get_parameter_value(parameters, "port_gap")
            )
        design_data.append(this_design_data)
        
    if SOLVE_MODE == "HFSS":
        solve = call_solver
    elif SOLVE_MODE == "DUMMY":
        solve = call_solver_dummy
    else:
        raise KeyError(f"Unknown SOLVE_MODE: {SOLVE_MODE}")

    with Pool(processes=MAX_PARALLEL_SOLVE_PROCESSES) as pool:
        results = []
        for result in tqdm(pool.imap(solve, design_data), total=len(design_data)):
            results.append(result)

    for design, result in zip(design_data, results):
        result_design = {}
        #print(f"design[0]: {design[0]}, result: {result}")
        result_design["hid"] = design[0]
        responses = [{"name": "return_loss", "value": result}]
        result_design["responses"] = responses
        result_design_list.append(result_design)

    print(f"Return {len(result_design_list)} designs")
    return result_design_list


def add_system(
    system_name: str,
    parent_system: ParametricSystem,
    num_designs_max: int = 100,
    ):
    system: ParametricSystem = parent_system.create_node(type_=node_types.Sensitivity, name="Sensitivity")

    # Modify algorithm settings
    settings = system.get_property("AlgorithmSettings")
    settings["num_discretization"] = num_designs_max
    system.set_property("AlgorithmSettings", settings)

    # Fast running solver settings
    #system.set_property("AutoSaveMode", "no_auto_save")
    #system.set_property("SolveTwice", True)
    #system.set_property("UpdateResultFile", "at_end")
    # system.set_property("WriteDesignStartSetFlag", False)

    return system


def add_solver_node_to_parent_system(parent_system):
    proxy_solver: ProxySolverNode = parent_system.create_node(
        type_=node_types.ProxySolver, name="MyProxySolver", design_flow=DesignFlow.RECEIVE_SEND
    )
    multi_design_launch_num = -1  # set -1 to submit all designs simultaneously
    proxy_solver.set_property("MultiDesignLaunchNum", multi_design_launch_num)
    proxy_solver.set_property("ForwardHPCLicenseContextEnvironment", True)
    return proxy_solver


def add_mop_node(parent_system, predecessor_system):
    mop : Node = parent_system.create_node(
            type_=node_types.Mop, name="MOP"
            )

    # connect
    predecessor_system.get_output_slots("OMDBPath")[0].connect_to(
        mop.get_input_slots("IMDBPath")[0]
    )
    predecessor_system.get_output_slots("OParameterManager")[0].connect_to(
        mop.get_input_slots("IParameterManager")[0]
    )


    """
    props = mop.get_properties()
    info = mop._get_info()
    mop_custom_settings = mop.get_property("MOPCustomSettings")
    print(mop_custom_settings)
    signal_competition = mop_custom_settings["competition_signal_model"]["sequence"][2]["Second"]
    signal_competition[0]["Used"] = True  #Baseline model
    signal_competition[1]["Used"] = True  #PCA based
    signal_competition[2]["Used"] = False #DIMGP Signal
    signal_competition[3]["Used"] = False #LSTM
    signal_competition[4]["Used"] = True  #IndexNN
    mop.set_property("MOPCustomSettings", mop_custom_settings)
    """
    return mop


def prepare_parameter_response_schema(parameters, responses):
    load_json = {}
    load_json["parameters"] = []
    load_json["responses"] = []
    for parameter_name, parameter_data in parameters.items():
        parameter = {"dir": {"value": "input"}, "name": parameter_name, "value": parameter_data["reference_value"]}
        load_json["parameters"].append(parameter)

    for response_name, response_data in responses.items():
        response = {"dir": {"value": "output"}, "name": response_name}
        load_json["responses"].append(response)

    return load_json

def main(osl_project_name, parameters, responses, num_designs_max=100):
    osl = Optislang(project_path=str(WORKING_DIR / osl_project_name))
    root_system = osl.application.project.root_system
    system = add_system("Sensitivity", root_system, num_designs_max=num_designs_max)

    proxy_solver = add_solver_node_to_parent_system(system)

    proxy_solver.load(args=prepare_parameter_response_schema(parameters, responses))

    proxy_solver.register_locations_as_parameter()
    proxy_solver.register_locations_as_response()

    # Change parameter bounds.
    for parameter_name, parameter_data in parameters.items():
        system.parameter_manager.modify_parameter(
            OptimizationParameter(name=parameter_name, 
                reference_value=parameter_data["reference_value"],
                range=(parameter_data["lower_bound"], parameter_data["upper_bound"])
                )
            )
            
    osl.application.save()
    osl.application.project.start(wait_for_finished=False)

    all_responses = []
    while True:
        time.sleep(1.0)
        print("Waiting for optiSLang")
        # Get a batch of design points
        design_list = proxy_solver.get_designs()
        # Pass the designs points to the compute_designs function
        responses_list = compute_designs(design_list)
        # collect the responses
        all_responses += responses_list
        # return responses back to the ProxySolver object
        proxy_solver.set_designs(responses_list)
        if len(all_responses) == num_designs_max:
            break

    print("Parametric design study done!")
    
    mop = add_mop_node(root_system, system)
    print("Added and connected MOP node.")

    print("Continue workflow execution (this runs the newly added MOP node to train surrogate models based on the computed designs and responses)...")
    osl.application.project.start()

    print("Solved Successfully!")

    for response in all_responses:
        freq = response["responses"][0]["value"]["abscissa"]
        return_loss = response["responses"][0]["value"]["channels"][0]
        plt.plot(freq, return_loss, label=response["hid"])

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Return loss [dB]")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    osl_project_name = "pyoptislang_example_proxy_solver.opf"
    parameters = {
            "l_dipole": {"reference_value": 10.2, "lower_bound": 9.0, "upper_bound": 12.0},
            "wire_rad": {"reference_value": 1, "lower_bound": 0.8, "upper_bound": 1.2},
            "port_gap": {"reference_value": 1, "lower_bound": 0.8, "upper_bound": 1.2},
            }
    responses = {
            "return_loss": {}
            }

    num_designs_max = 20
    main(osl_project_name, parameters, responses, num_designs_max=num_designs_max)
