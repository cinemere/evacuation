import os

MODELS_DIR = "/home/rice/aklepach/code/evacuation/saved_data/models"

FULLENSL_DIR = "n60_grademb_exitrew_followrew_intrrew-0._initrew--1._alpha-[2,3,4,5]_noise-.5_ensldegree-1."
FULLENSL_FILE_1 = "n60_grademb_exitrew_followrew_intrrew-0._initrew--1._alpha-2_noise-.5_ensldegree-1._n-60_lr-0.0003_gamma-0.99_s-gra_a-2.0_ss-0.01_vr-0.1_23-Nov-04-47-27.zip"

FULLENSL_MODEL_1 = os.path.join(MODELS_DIR, FULLENSL_DIR, FULLENSL_FILE_1)