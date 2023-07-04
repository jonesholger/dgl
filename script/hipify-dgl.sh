for fle in $(find . -name *.cu )
do
# some fancy logic to change the path something/cuda/something -> /something/hip/something
output_fle=$(sed -e 's/\/cuda\//\/hip\//' -e 's/cu$/cpp/' <<< $fle)
info_fle=${output_fle/%cpp/info}
echo $fle $output_fle $info_fle
if [ ! -d $(dirname $output_fle) ]
then
mkdir -p $(dirname $output_fle)
fi
hipify-perl -hip-kernel-execution-syntax -print-stats -o ${output_fle} $fle 2>${info_fle}
done

for fle in $(find . -name *.cuh )
do
# some fancy logic to change the path something/cuda/something -> /something/hip/something
output_fle=$(sed -e 's/\/cuda\//\/hip\//' -e 's/cuh$/h/' <<< $fle)
info_fle=${output_fle/%h/info}
echo $fle $output_fle $info_fle
if [ ! -d $(dirname $output_fle) ]
then
mkdir -p $(dirname $output_fle)
fi
hipify-perl -hip-kernel-execution-syntax -print-stats -o ${output_fle} $fle 2>${info_fle}
done
