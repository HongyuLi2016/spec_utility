#Spectrum fitting utilities 
========

Python scripts for spectrum fitting with MaNGA data (not limitied to MaNGA) based on starlight and pPXF software. 
Include data preparation, software wrapper, output analysis and visulization.

========

Contents
-----
Below is a brief description of the contents of the directories in the root directory:

 * `MaNGA`: Contains the python scripts handling with MaNGA data preparation

 * `data`: Contains some useful data files (e.g. templates properties)

 * `ppxf`: Contains the python wrapper for ppxf
 
 * `starlight`: Contains the python scripts for starlight input file preparation, output analysis and visulization.
 
 * `test`: Contains some test scripts and data
 
 * `utils`: other utilities for ploting figures, dumping/loading data etc.
 
Workflow
-----
 * `starlight`
  * `make_ssp_data.py`: MaNGA data preparation. 
    This will create a folder named spec in the working directory, which contain the txt spectrum files and vorinoi
    bin information.
  * `create_grid.py`: Starlight input grid file preparation.
    This will create 3 folders (grid, grid files; juck, pbs *.o *.e files; out, starlight output resutls). The grid
    file will created in grid folder.
  * `pbs_starlight.sh`: Pbs script preparation.
  * `copy.sh` (under ~/STARLIGHTv04/config/): Copy some base files and configuration files under the working directory.
  * `qsub.py`: submit pbs jobs
  * `dump_starlight.py`: dump starlight output files into a binary file (pickable python dictionary)
  * `ssp_maps.py`: create a map file (e.g. M*/L, logAge, [Z/H]) in fits format.
  * `sps_plot_spec.py`: plot figures if necessary. 
  
 * `ppxf`
  * `make_ssp_data.py`: MaNGA data preparation. (This is not necessary if one has run this for starlight)
    This will create a folder named spec in the working directory, which contain the txt spectrum files and vorinoi
    bin information.
  * `pbs_ppxf.sh`: Pbs script preparation.
  * `qsub.py`: submit pbs jobs. One can also choose to run without pbs (using run_ppxf.py)
  * `ssp_maps.py`: create a map file (e.g. M*/L, logAge, [Z/H]) in fits format.
  * `download_ssp.sh`: create a *_rst folder in the working directory, including dumped data and some necessary files.
  * `sps_plot_spec.py`: plot figures if necessary.

  
working directory structure
-----
 * `gname`: Root directory for a galaxy. At least contain the information.fits file (provide necessary information 
   about the galaxy like redshift)
 * `gname/manga-plate-ifu-LINECUBE.fits.gz`: MaNGA DRP line cube. Not necessary, if -p option is set in make_ssp_data.py
 * `gname/spec`: Contains model spectra in txt format
 * `gname/grid`: Contains starlight input grid files
 * `gname/out`: Contains starlight output files
 * `gname/juck`: Contains pbs *.e/*.o files
 * `gname/pbsscript`: Contains pbs script for job submission
 * `gname/dump`: Contains the binery files dumped by ppxf (pickable python dictionary)
 * `gname/dump_star`: Same dumped files for starlight
 * `gname/gname_rst`: Dumped binery files and other necessary files to be donwloaded.
 
