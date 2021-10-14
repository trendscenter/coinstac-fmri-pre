FROM trendscenter/aa-fmri-spm-tpm-epi:v1.0.0_20210820

# FSL installer appears to now be ready for use with version 6
# eddy is also now included in FSL6
RUN wget -q http://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py && \
    chmod 775 fslinstaller.py && \
    python /fslinstaller.py -d /opt/fsl -V 6.0.4 -q && \
    rm -f /fslinstaller.py
RUN which immv || ( echo "FSLPython not properly configured; re-running" && rm -rf /opt/fsl/fslpython && /opt/fsl/etc/fslconf/fslpython_install.sh -f /opt/fsl || ( cat /tmp/fslpython*/fslpython_miniconda_installer.log && exit 1 ) )
RUN wget -qO- "https://www.nitrc.org/frs/download.php/5994/ROBEXv12.linux64.tar.gz//?i_agree=1&download_now=1" | \
    tar zx -C /opt


    
#-------------------------------------------------
# Set environment variables
#-------------------------------------------------
ENV FSLDIR=/opt/fsl
ENV PATH=${FSLDIR}/bin
RUN source ${FSLDIR}/etc/fslconf/fsl.sh
ENV FSLOUTPUTTYPE=NIFTI_GZ  


# Copy the current directory contents into the container
COPY . /computation

RUN groupadd --gid 1000 node \
  && useradd --uid 1000 --gid node --shell /bin/bash --create-home node




ENV YARN_VERSION 1.6.0



ADD server/. /server
WORKDIR /server
RUN npm i --production
CMD ["node", "/server/index.js"]

