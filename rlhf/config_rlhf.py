# The KL penalty coefficient
# If you set this too high the agent will be penalized too much for deviating from the original policy.

import os

# This looks for an environment variable 'RLHF_BETA'. 
# If it doesn't find it, it defaults to 0.1.
BETA = float(os.getenv("RLHF_BETA", 0.1))