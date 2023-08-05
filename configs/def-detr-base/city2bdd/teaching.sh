N_GPUS=2
BATCH_SIZE=8
DATA_ROOT=<YOUR/DATA/ROOT>
OUTPUT_DIR=./outputs/def-detr-base/city2bdd/teaching

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26508 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 9 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset cityscapes \
--target_dataset bdd100k \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-5 \
--lr_backbone 2e-6 \
--lr_linear_proj 2e-6 \
--epoch 20 \
--epoch_lr_drop 20 \
--mode teaching \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/../cross_domain_mae/model_best.pth \
--epoch_retrain 10 \
--epoch_mae_decay 5 \
--alpha_dt 0.9 \
--max_dt 0.6 \
--teach_box_loss True

