for path in {1..19}; do
    python inference.py --checkpoint_path ./pretrained/model.pth --device 0 --events_file_path ./eval_data/$path.h5 --output_folder ./reconstruction/$path/ --loader_type CH5
done