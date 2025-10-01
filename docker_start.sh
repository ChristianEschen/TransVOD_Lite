sudo docker build --no-cache -f docker/Dockerfile --progress=plain -t detr .
docker run --rm -it \
  --network host \
  --ipc=host \
  --rm --gpus all \
  -v /home/christianeschen:/home/christianeschen \
  -w /home/christianeschen/transvod_lite_docker/TransVOD_Lite \
  detr
pip install -r requirements.txt
cd models/ops
bash make.sh

cd ../..

python -u main.py \
    --lr 1.25e-05 \
    --backbone swin_t_p4w7 \
    --lr_backbone 1.25e-06 \
    --epochs 7 \
    --lr_linear_proj_mult 0.00625 \
    --lr_drop_epochs 5 6 \
    --num_feature_levels 1\
    --num_queries 100 \
    --dilation \
    --batch_size 2 \
    --hidden_dim 256 \
    --num_workers 8 \
    --with_box_refine \
    --coco_pretrain \
    --dataset_file 'vid_single' \
    --vid_path /home/christianeschen/new_ml/swt_ml_training_pipeline/export \
    --output_dir "my_exp_dir_lr_red_16_7ep"
