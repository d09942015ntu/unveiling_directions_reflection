from collections import defaultdict
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

def parse_json_file(json_file_expr):
    json_files_paths = sorted(glob.glob(json_file_expr))
    acc = None
    for json_files_path in json_files_paths:
        json_item = json.load(open(json_files_path))
        for t_name in json_item['results'].keys():
            if 'exact_match,flexible-extract' in  json_item['results'][t_name]:
                acc = json_item['results'][t_name]['exact_match,flexible-extract']
                break
            elif "pass@1,create_test" in json_item['results'][t_name]:
                acc = json_item['results'][t_name]["pass@1,create_test"]
                break

    assert acc is not None
    return acc

def png_result(plot_data, plot_meta):
    plt.clf()
    plt.figure(figsize=(6,4))
    for plot_data_i in plot_data:
        plt.plot(plot_data_i["x"],
                 plot_data_i["y"],
                 label=plot_data_i["label"],
                 linestyle=plot_data_i["linestyle"]
                 )
    plt.legend(fontsize=14)
    plt.ylim(0,plot_meta["ymax"])
    plt.xlabel("$\\ell$")
    plt.ylabel("Accuracy")
    plt.title(plot_meta["title_png"], fontsize=16)
    plt.savefig(plot_meta["fig_name_png"], bbox_inches="tight")

def tikz_result(plot_data, plot_meta):
    out_str = ""
    out_str += ("""
    \\begin{tikzpicture}
    \\begin{axis}[
        align=center,
        width=3.8cm,
        height=3cm,
        xlabel={$\\ell$},
        ylabel={Acc},
        legend pos=north west,
        grid=major,
        grid style={dashed,gray!30},
        xmin=0, xmax=%s,
        ymin=-0.05, ymax=%s,
        title={%s},
        title style={font=\\scriptsize},
        label style={font=\\scriptsize},
        tick label style={font=\\tiny},
        legend style={font=\\tiny},
        legend pos=outer north east,
        legend cell align={left},
        xlabel style={
            at={(current axis.south east)}, 
            anchor=north east,
            yshift=10pt,
            xshift=5pt
        },
        ylabel style={
            at={(current axis.north west)}, 
            anchor=north east,
            yshift=-20pt,
            xshift=20pt
        },
        ytick={%s}
    ]
    """ % (plot_meta["xmax"], plot_meta["ymax"], plot_meta["title_tikz"].replace("_","\_"),
           "0,0.2,0.4,0.6,0.8,1" if plot_meta["ymax"] > 0.5 else "0,0.1,0.2,0.3,0.4,0.5"))
    for i, plot_data_i in enumerate(plot_data):
        thickness="thick"
        if plot_data_i["linestyle"] == "dotted":
            thickness="very thick"
        out_str += ("\\addplot[%s, %s, %s] table[row sep=\\\\] {\n" % (plot_data_i["color_tikz"], thickness, plot_data_i["linestyle"]))
        out_str += ("  x y \\\\ \n")
        for x,y in zip(plot_data_i["x"],plot_data_i["y"]):
            out_str += (f" {x} {y} \\\\ \n")
        out_str += ("}; \n")
        #f.write("\\addlegendentry{%s}" % plot_data_i["label"])
    out_str += ("""\\end{axis} \n \\end{tikzpicture} \n
    """)
    with open(plot_meta["fig_name_tikz"], "w") as f:
        f.write(out_str)
    with open(plot_meta["fig_name_tikz_2"], "w") as f:
        f.write(out_str)




def parse_all_result(dataset_name, input_dir, output_dir, ymax, limit, model_name):

    if model_name == "qwen3b":
        skey_stype={
            1:["eos","answer"],
            -1: ["wait","eos"],
        }
    elif model_name == "gemma4b":
        skey_stype={
            1:["<eos>","answer"],
            -1: ["wait","<eos>"],
        }
    else:
        assert 0

    model_map={
        "qwen3b":"Qwen2.5-3B",
        "gemma4b":"Gemma3-4B"
    }

    max_layers = {
        "qwen3b": 35,
        "gemma4b": 33
    }

    for s in skey_stype.keys():
        if s == 1:
            steering_vector_name={
                "20_tselected": "Level2 - Level0",
                "21_tselected": "Level2 - Level1",
            }
        elif s == -1:
            steering_vector_name={
                "20_tselected": "Level2 - Level0",
                "10_tselected": "Level1 - Level0",
            }
        else:
            assert 0
        steering_type = list(steering_vector_name.keys())

        for start_type in skey_stype[s]:
            plot_data = []
            plot_meta = {}
            plt.clf()
            plt.figure(figsize=(8, 4))
            ax = plt.subplot(111)
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            random_dict = defaultdict(list)
            layer_ids = list(range(max_layers[model_name]+1))
            i=0
            for steering_type in steering_type:
                results = []
                layer_id_exists = []
                for layer_id in layer_ids:
                    json_files_expr = os.path.join(input_dir,f"s{s}_{start_type}_{limit}_{steering_type}_l{layer_id}", ".*", "*.json")
                    print(json_files_expr)
                    json_files_paths = sorted(glob.glob(json_files_expr))
                    if len(json_files_paths) >= 1:
                        layer_id_exists.append(layer_id)
                        acc = parse_json_file(json_files_paths[0])
                        results.append(acc)
                print(f"layer_ids={layer_ids}, results={results}, steering_type={steering_type}")
                if len(layer_id_exists) > 0:
                    plot_data.append({
                        "x": layer_id_exists,
                        "y": results,
                        "label": f"{steering_vector_name[steering_type]}",
                        "linestyle": "solid",
                        "color_tikz": f"c{11+i}",
                    })
                    i += 1

            gt_groups = {
                "L2": ["Wait", "Alternatively", "Check"],
                "L1": ["<|endoftext|>", "#", "%", "<eos>"],
                "L0": ["Answer", "Result", "Output"],
            }
            for gkey, gwords in gt_groups.items():
                acc = []
                for gt_type in gwords:
                    json_files_expr = os.path.join(input_dir.replace("step4", "step1"), f"gt_{limit}_{gt_type}",
                                                   ".*", "*.json")
                    json_files_paths = sorted(glob.glob(json_files_expr))
                    if len(json_files_paths) > 0:
                        acc.append(parse_json_file(json_files_paths[0]))
                plot_data.append({
                    "x": [0, max_layers[model_name]],
                    "y": [np.average(acc)] * 2,
                    "label": f"baseline: {gkey}",
                    "linestyle": "dotted",
                    "color_tikz": f"c{11 + i}"
                })
                i += 1

            gt_type_starts ={
                "wait":["Wait"],
                "eos":["<eos>", "<|endoftext|>"],
                "<eos>":["<eos>", "<|endoftext|>"],
                "answer":["Answer"]
            }

            for gt_type in gt_type_starts[start_type]:
                json_files_expr = os.path.join(input_dir.replace("step4","step1"), f"gt_{limit}_{gt_type}", ".*", "*.json")
                json_files_paths = sorted(glob.glob(json_files_expr))
                if len(json_files_paths) >= 1:
                    acc = parse_json_file(json_files_paths[0])

                    plot_data.append({
                        "x": layer_ids,
                        "y": [acc]*len(layer_ids),
                        "label": f"gt:{gt_type}",
                        "linestyle": "dotted",
                        "color_tikz": f"c{11 + i}",
                    })


            startype_title={
               "wait" :"Wait",
                "eos" :"[EOS]",
                "<eos>":"[EOS]",
                "answer":"Answer",
            }[start_type]
            plot_meta["title_png"] = f"{model_map[model_name]},\\\\ {dataset_name},\\\\ Intervene `{startype_title}'"
            plot_meta["title_tikz"] = f"{model_map[model_name]},\\\\ {dataset_name},\\\\ Intervene `{startype_title}'"
            plot_meta["fig_name_png"] = os.path.join(output_dir, f"png_exp4_{model_name}_{dataset_name}_{s}_{start_type.replace("<eos>","eos")}.png")
            plot_meta["fig_name_tikz"] = os.path.join(output_dir, f"exp4_{model_name}_{dataset_name}_{s}_{start_type.replace("<eos>","eos")}.tex")
            plot_meta["fig_name_tikz_2"] = os.path.join("all_exp4_latex_fig", f"exp4_{model_name}_{dataset_name}_{s}_{start_type.replace("<eos>","eos")}.tex")
            plot_meta["ymax"] = ymax
            plot_meta["xmax"] = max_layers[model_name]

            png_result(plot_data, plot_meta)
            tikz_result(plot_data, plot_meta)




def run():
    dataset_name="gsm8k_adv"
    input_dir="visualize/gsm8k_adv/qwen3b/step4"
    output_dir="visualize/gsm8k_adv/qwen3b/step6"
    ymax=0.6
    limit = 2000
    model_name="qwen3b"
    os.makedirs(output_dir, exist_ok=True)

    parse_all_result(dataset_name, input_dir, output_dir, ymax, limit, model_name)



if __name__ == '__main__':
    run()