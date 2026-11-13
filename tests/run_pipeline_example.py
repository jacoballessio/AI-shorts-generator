import os, sys
# ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline_orchestrator import PipelineOrchestrator

if __name__ == '__main__':
    po = PipelineOrchestrator(base_dir='data')
    po.process_video('https://www.youtube.com/watch?v=sENByrMHfyk')
