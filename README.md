#dir structure
gname/
gname/linecube (not necessary, if make_ssp_data.py set -p option)
gname/information.fits redshift
#workflow
make_ssp_data.py: linecube => txt spectrum
##ppxf
run_ppxf.py => gname/dump and gname/figs (if -p)
##starlight
create_grid.py => gname/grid  gname/out
pbs_starlight.sh => gname/pbsscript
qsub.py -g gname qsub  => run starlight
dump_starlight.py  => gname/dump_star and gname/figs_star (if -p)
##extract maps
ssp_maps.py 284_8082-12701 -f dump_star
##download dump and maps
download_ssp.sh 284_8082-12701
##plot figures if necessary
sps_plot_spec.py dumped_file_name -s (-s for save as a png fig)
