names=(
	# maize_yield
	# rice_yield
	# wheat_yield
	# soy_yield
	# daily_temperature_variability
	# number_of_wet_days
	# extreme_daily_rainfall
	# rx5day
	# wet_bulb_temperature
	# river_discharge
	# runoff
	tas
	pr
	tasmax
	tasmin
	sfcwind
	hurs
	huss
	ps
	prsn
	rlds
	rsds
)

# rime-download-isimip -v tas
# rime-pre-gmt
# rime-pre-wl

for name in ${names[@]}; do
	echo ">>>$name<<<"
	weight=latWeight
	rime-download-isimip -i $name
	rime-pre-region --weight $weight -i $name --cpus 20
	rime-pre-quantilemap --weight $weight -i $name --regional-no-admin --regional --map --equi --map-chunk 90
done

rime-pre-digitize -i ${names[@]} --weights latWeight
rime-pre-digitize -i ${names[@]} --weights latWeight --all-subregions