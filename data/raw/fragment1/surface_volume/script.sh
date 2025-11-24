BASE="https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface/surface_volume"

for i in $(seq -w 0 30); do
    echo "Downloading $i.tif ..."
    wget "$BASE/$i.tif"
done
