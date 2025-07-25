graph_dirs = ['soda-benchmark-graphs/pytorch-graphs/','soda-benchmark-graphs/tflite-graphs/','test-data/']
graph_files = [['mobile_net_tosa.dot','rez_net_tosa.dot','squeeze_net_tosa.dot'],
['anomaly_detection_tosa.dot','image_classification_tosa.dot',
'visual_wake_words_tosa.dot','keyword_spotting_tosa.dot'],
['01_tosa.dot']]

## fixed hw scale factor 0.3
## fixed seed 42
if __name__ == "__main__":
    for gdiridx,gdir in enumerate(graph_dirs):
        for graph_name in graph_files[gdiridx]:
            for area_constraint in [0.1,0.3,0.5,0.9]: ## area constraint
                for l in [0.5,1,5,10]: ## hw scale variance
                    for mu in [0.5,1,5,10]: ## communication scale
                        print(f'sbatch pso_eval.sbatch {gdir}{graph_name} {area_constraint} 0.3 {l} {mu} 42')