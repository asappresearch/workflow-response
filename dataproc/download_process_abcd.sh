folder="data/abcd/raw"
mkdir -p $folder

json_file="${folder}/abcd_v1.1.json"
gz_file="${json_file}.gz"

# download & unzip raw file
if [ -f $json_file ]; then
    echo "${json_file} exists. Skip download."
else
    if [ -f $gz_file ]; then
        echo "${gz_file} exists. Skip download."
    else
        wget https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz -O $gz_file
    fi
    gzip -d $gz_file
fi

# create base model datasets
base_folder="${folder/raw/base}"
python dataproc/make_seed_data.py 
