# Only need to change these two variables
PKG_NAME=bw_calc-dev
USER=cmutel

mkdir ~/conda-bld
conda config --set anaconda_upload no
conda config --add channels haasad
conda config --add channels conda-forge
export CONDA_BLD_PATH=~/conda-bld
export VERSION=`date +%Y.%m.%d`
conda build . --old-build-string
ls $CONDA_BLD_PATH/noarch/
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l nightly $CONDA_BLD_PATH/noarch/$PKG_NAME-$VERSION-py_0.tar.bz2 --force
