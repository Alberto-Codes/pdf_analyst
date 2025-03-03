import pytest
import sys
import os
import inspect

def test_import_paths():
    """Print the sys.path and content of src directory."""
    print(f"Python path: {sys.path}")
    
    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
    if os.path.exists(src_dir):
        print(f"Contents of {src_dir}:")
        for item in os.listdir(src_dir):
            print(f"  - {item}")
    
    # Try to import pydantic_graph and see what's available
    try:
        import pydantic_graph
        print("\nContents of pydantic_graph module:")
        for name in dir(pydantic_graph):
            if not name.startswith('_'):
                print(f"  - {name}")
                
        # Check what's in the graph module
        if hasattr(pydantic_graph, 'graph'):
            print("\nContents of pydantic_graph.graph module:")
            graph_module = pydantic_graph.graph
            for name in dir(graph_module):
                if not name.startswith('_'):
                    print(f"  - {name}")
                    
            # Check if Graph class has run method
            if hasattr(graph_module, 'Graph') and hasattr(graph_module.Graph, 'run'):
                print("\nGraph.run method found in pydantic_graph.graph")
                
        # Check what's in other modules
        for module_name in ['executor', 'context', 'state', 'nodes']:
            if hasattr(pydantic_graph, module_name):
                print(f"\nContents of pydantic_graph.{module_name} module:")
                module = getattr(pydantic_graph, module_name)
                for name in dir(module):
                    if not name.startswith('_') and 'run' in name.lower():
                        print(f"  - {name}")
        
        # Look for run_graph in all modules
        print("\nSearching for run_graph function...")
        for module_name in dir(pydantic_graph):
            if module_name.startswith('_'):
                continue
                
            module = getattr(pydantic_graph, module_name)
            if hasattr(module, 'run_graph'):
                print(f"run_graph found in pydantic_graph.{module_name}")
            
            # Look in submethods too
            if inspect.ismodule(module):
                for subname in dir(module):
                    if subname.startswith('_'):
                        continue
                    subattr = getattr(module, subname)
                    if hasattr(subattr, 'run_graph'):
                        print(f"run_graph found in pydantic_graph.{module_name}.{subname}")
                    elif callable(subattr) and 'run_graph' == subname:
                        print(f"run_graph function found in pydantic_graph.{module_name}.{subname}")
    
    except ImportError as e:
        print(f"Error importing pydantic_graph: {e}")
        
if __name__ == "__main__":
    test_import_paths() 