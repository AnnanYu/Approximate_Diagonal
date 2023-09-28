XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 USE_WANDB=0 python run_train.py \
 --C_init=lecun_normal --batchnorm=True --bidirectional=True --blocks=3 --bsz=50 \
 --clip_eigs=True --conj_sym=False --d_model=512 --dataset=lra-cifar-classification \
 --epochs=250 --jax_seed=16418 --lr_factor=5 --n_layers=6 --opt_config=BfastandCdecay \
 --p_dropout=0.1 --ssm_lr_base=0.001 --ssm_size_base=384 --warmup_end=1 --weight_decay=0.07