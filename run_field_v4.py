"""Convenience entry point for the corrected field workflow."""
import sys
from run_full import main
if __name__=='__main__':
    if '--stage' in sys.argv:raise SystemExit('run_field_v4 always runs field; omit --stage')
    sys.argv.extend(['--stage','field']);main()
