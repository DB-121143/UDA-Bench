USE_BART = False

BATCH_SIZE = 15
BUDGET = 0
# without filter sampling
# small sample
# BUDGET = 0
# BATCH_SIZE = 10

# aggregation strategy
AGGR_STRATEGY = 'normal'
# AGGR_STRATEGY = 'skip-filter'

# cluster parameters
N_CENTROIDS = 6
N_ITER = 40
# Sampling ratio for extracting classification labels
GROUP_EXTRACT_SAMPLE_RATIO = 0.2
# Sampling ratio in each clustering core
AGGR_CLUSTER_SAMPLE_RATIO = 0.6

MODEL = "gpt-4.1-mini"
OPENAI_KEY = "your-key"
BASE_URL = "your-base-url"
