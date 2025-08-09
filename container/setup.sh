#!/bin/sh

INSTALL_DIR=$HOME/nerproj_img

#LOCAL_REPO="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
LOCAL_REPO="/aloy/home/ymartins/match_clinical_trial/container/"

LOCAL_IMAGE=$INSTALL_DIR/nermatchct_img.simg

LOCAL_IMAGE_SANDBOX=$INSTALL_DIR/sandbox


# compare versions
version_gt () {
    test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1";
}


# check if singularity is available
_=$(command -v singularity);
if [ "$?" != "0" ]
then
    printf -- "\033[31m ERROR: You don't seem to have Singularity installed \033[0m\n";
    printf -- 'Follow the guide at: https://www.sylabs.io/guides/2.6/user-guide/installation.html\n';
    exit 1;
fi;

# check singularity version
SINGULARITY_MIN_VERSION=2.5.0
SINGULARITY_INSTALLED_VERSION="$(singularity --version)"
if version_gt $SINGULARITY_MIN_VERSION $SINGULARITY_INSTALLED_VERSION
then
    printf -- "\033[31m ERROR: Update Singularity, we require at least version ${SINGULARITY_MIN_VERSION} (${SINGULARITY_INSTALLED_VERSION} detected) \033[0m\n";
    printf -- 'Follow the guide at: https://www.sylabs.io/guides/2.6/user-guide/installation.html\n';
    exit 2;
fi

if [ -d "$INSTALL_DIR" ]
then
    printf -- '\033[33m WARNING: %s already exists, delete it to proceed. \033[0m\n' $INSTALL_DIR;
    exit 0;
fi
mkdir $INSTALL_DIR;
cd $INSTALL_DIR;
SINGULARITY_DEFINITION="$LOCAL_REPO/nermatchct.def"

cp $SINGULARITY_DEFINITION $INSTALL_DIR;
mkdir -p ./container/

ENVCONFIG="$LOCAL_REPO/nerworkflow.yml"
cp -r $ENVCONFIG ./container/;
 
printf -- 'Removing old singularity image...\n';
sudo rm -f $LOCAL_IMAGE;
sudo rm -rf $LOCAL_IMAGE_SANDBOX;

printf -- 'Creating singularity sandbox image... \n';
sudo singularity build --sandbox $LOCAL_IMAGE_SANDBOX $SINGULARITY_DEFINITION  #cc_py37.def;
if [ $? -eq 0 ]; then
    printf -- '\033[32m SUCCESS: Image sandbox created correctly. \033[0m\n';
else
    printf -- '\033[31m ERROR: Cannot create sandbox image. \033[0m\n';
    exit 5;
fi

# generate image from sandbox
sudo singularity build $LOCAL_IMAGE $LOCAL_IMAGE_SANDBOX;
if [ $? -eq 0 ]; then
    printf -- '\033[32m SUCCESS: Image file created. \033[0m\n';
else
    printf -- '\033[31m ERROR: Cannot create image. \033[0m\n';
    exit 6;
fi


