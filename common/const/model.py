""" Encoder configurations """
MDL_ENCODER = 'encoder'  #: ENCODER KEY
DEF_ENCODER = 'google/electra-base-discriminator'  #: DEFAULT ENCODER VALUE

""" Decoder configurations """
MDL_EQUATION = 'equation'  #: DECODER KEY
MDL_Q_PATH = 'path'  #: PATH TO LOAD DECODER
MDL_Q_ENC = 'encoder_config'  #: ENCODER CONFIGURATION
MDL_Q_EMBED = 'embedding_dim'  #: EMBEDDING DIMENSION
MDL_Q_HIDDEN = 'hidden_dim'  #: HIDDEN STATE DIMENSION
MDL_Q_INTER = 'intermediate_dim'  #: INTERMEDIATE LAYER DIMENSION
MDL_Q_LAYER = 'layer'  #: NUMBER OF HIDDEN LAYERS
MDL_Q_INIT = 'init_factor'  #: INITIALIZATION FACTOR FOR TRANSFORMER MODELS
MDL_Q_LN_EPS = 'layernorm_eps'  #: EPSILON VALUE FOR LAYER NORMALIZATION
MDL_Q_HEAD = 'head'  #: NUMBER OF MULTI-ATTENTION HEADS
MDL_DECREMENTER = 'decrementer' #: DECREMENT FACTOR OF GOLD TEXT BEING COPIED

""" Keyword Selector model configurations """
MDL_KEYWORD = 'keyword' # KEYWORD MODEL KEY
MDL_K_SHUFFLE_ON_TRAIN = 'shuffle'
LOSS_KL_PRIOR = 'kl_prior' #: PRIOR FOR KL-DIVERGENCE CONSTRAINT (LOSS HYPERPARAMETER)
LOSS_KL_COEF = 'kl_coefficient' #: COEFFICIENT OF THIS CONSTRAINT (LOSS HYPERPARAMETER)

""" Decoder configuration default """
DEF_Q_EMBED = 128  #: FALLBACK VALUE FOR EMBEDDING DIM
DEF_Q_HIDDEN = 768  #: FALLBACK VALUE FOR HIDDEN DIM
DEF_Q_INTER = 2048  #: FALLBACK VALUE FOR INTERMEDIATE DIM
DEF_Q_LAYER = 6  #: FALLBACK VALUE FOR NUMBER OF HIDDEN LAYERS
DEF_Q_INIT = 0.02  #: FALLBACK VALUE FOR INITIALIZATION FACTOR
DEF_Q_LN_EPS = 1E-8  #: FALLBACK VALUE FOR LAYER NORMALIZATION EPSILON
DEF_Q_HEAD = 12  #: FALLBACK VALUE FOR NUMBER OF MULTI-HEAD ATTENTIONS
