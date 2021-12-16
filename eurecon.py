import os.path as path
import sys

import click

base_dir = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(base_dir)

from eurecon import Eurecon

@click.command()
@click.option('--rmsd', '-r', help = 'Desired RMSD parameter')
@click.option('--partition', '-p', help = 'Desired partition parameter; 0 - translation only, 1 - rotation only')
@click.option('--axes_path', '-a', help = 'Path to the file, containing tessellation axes')
@click.option('--input_directory', '-i', help = 'Input directory for the base file')
@click.option('--output_directory', '-o', help = 'Output directory for the generated conformations')
@click.option('--stdout_mode', '-s', help = 'Enables/disables resulting file generation', required = False)
@click.option('--weights_file', '-w', default = None, help = 'Path to the file, containing weights', required = False)
@click.option('--debug', '-d', help = 'Enables resulting RMSD check for the generated conformations', required = False)

def main(rmsd, partition, axes_path, input_directory, output_directory, stdout_mode, weights_file, debug):
    eurecon = Eurecon(
        input_directory,
        axes_path,
        output_directory,
        float(rmsd),
        float(partition),
        bool(stdout_mode),
        weights_file,
        debug
    )

    eurecon.start()


if __name__ == '__main__':
    main()
