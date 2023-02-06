# Quantization Range
quantize=1

# Quantization Range
q_qkv=1
q_ffn_1=1
q_ffn_2=1
q_emb=1
q_cls=1
layer_num=-1

# KD & Ternary Option
mean_scale=0.7
bert=base

#===========================================================#
# Logging Option
exp_name=TI_step1
neptune=0
save_quantized_model=1

# Distill Option
pred_distill=1
rep_distill=1
attn_distill=1
output_distill=1

# Teacher Intervention (TI)
teacher_attnmap=0
teacher_context=0
teacher_output=0
# TI-G options 
teacher_gradual=1
teacher_stochastic=0
teacher_inverted=0

# Training Type (downstream, qat_normal, qat_step1, qat_step2)
training_type=qat_normal

# DA Options
aug_train=0
aug_N=5

learning_rate=2E-5
# ===========================================================#

CUDA_VISIBLE_DEVICES=$1 python /home/ms/workspace/git/Teacher-Intervention-KD-QAT/main.py --data_dir data --task_name $2 --bert ${bert} \
--gpu 1 --quantize ${quantize} --qkv ${q_qkv} --ffn_1 ${q_ffn_1} --ffn_2 ${q_ffn_2} --emb ${q_emb} --cls ${q_cls} \
--aug_train ${aug_train} \
--output_distill ${output_distill} --pred_distill ${pred_distill} --rep_distill ${rep_distill} --attn_distill ${attn_distill} \
--teacher_attnmap ${teacher_attnmap} --teacher_context ${teacher_context} --teacher_output ${teacher_output} --teacher_gradual ${teacher_gradual} --teacher_stochastic ${teacher_stochastic} --teacher_inverted ${teacher_inverted} \
--training_type ${training_type} \
--mean_scale ${mean_scale} \
--exp_name ${exp_name} \
--save_quantized_model ${save_quantized_model} \
--neptune ${neptune} \
--aug_N ${aug_N} \
--num_train_epochs 3 --seed 1 \
--learning_rate ${learning_rate} 