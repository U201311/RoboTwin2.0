from . import transformers as transforms

# Make transforms available as a submodule
import sys
sys.modules[__name__ + '.transforms'] = transforms