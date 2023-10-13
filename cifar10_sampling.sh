DEVICES='0'


##########################


# CIFAR-10 (DDPM checkpoint) example

data="cifar10"
##data="imagenet"
sampleMethod='dpmsolver++'
type="dpmsolver"
steps="20"
DIS="logSNR"
order="3"
method="multistep"
workdir="experiments/"$data"/"$sampleMethod"_"$method"_order"$order"_"$steps"_"$DIS"_type-"$type

CUDA_VISIBLE_DEVICES=$DEVICES python certify.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --eta 0 --ni --skip_type=$DIS --sample_type=$sampleMethod --dpm_solver_order=$order --dpm_solver_method=$method --dpm_solver_type=$type --port 12350 

