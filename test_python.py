import sys
import torch
print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Test basic packages
print('\nTesting basic packages:')
packages = [
    'numpy', 'pandas', 'sklearn', 'nltk', 'textblob', 
    'transformers', 'networkx', 'matplotlib', 'tqdm',
    'gender_guesser'
]

for package in packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        print(f'✗ {package} - FAILED TO IMPORT')
        
# Test PyTorch Geometric
print('\nTesting PyTorch Geometric:')
try:
    import torch_geometric
    print(f'✓ torch_geometric {torch_geometric.__version__}')
    
    from torch_geometric.data import Data
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[1], [2], [3]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    print(f'✓ Can create a simple graph')
    print(f'  Data: {data}')
    
    # Try to import optional dependencies
    optional_deps = ['pyg_lib', 'torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_spline_conv']
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f'✓ {dep}')
        except ImportError:
            print(f'✗ {dep} - not installed (optional)')
            
except ImportError:
    print('✗ torch_geometric - FAILED TO IMPORT')

# Test FAISS
print('\nTesting FAISS:')
try:
    import faiss
    print(f'✓ faiss')
    dimension = 64
    nb = 100
    import numpy as np
    xb = np.random.random((nb, dimension)).astype('float32')
    index = faiss.IndexFlatL2(dimension)
    index.add(xb)
    print(f'✓ Can create a FAISS index')
except ImportError:
    print('✗ faiss - FAILED TO IMPORT')

print('\nTest completed.')