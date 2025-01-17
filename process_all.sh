names=(
	maize_yield
	rice_yield
	wheat_yield
	soy_yield
	daily_temperature_variability
	number_of_wet_days
	extreme_daily_rainfall
	rx5day
	wet_bulb_temperature
	river_discharge
	runoff
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

rime-download-isimip -v tas
rime-pre-gmt
rime-pre-wl

for name in ${names[@]}; do
	echo ">>>$name<<<"
	weight=latWeight
	# # weight=latWeight
	rime-download-isimip -i $name --remove-daily
	rime-pre-region --weight $weight -i $name --cpus 100
	rime-pre-quantilemap --weight $weight -i $name --regional-no-admin --regional --equi --skip-nans --cpus 200
	rime-pre-digitize -i $name --weights latWeight --cpus 200 --all-subregions
	rime-pre-quantilemap --weight $weight -i $name --map --equi --map-chunk 180 --skip-nans --warming-levels 1.0 1.2 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 --quantile-bins 11
done