FROM trendscenter/aa-fmri-spm-tpm-epi:v1.0.0_20210820

#-------------------------------------------------
# Set environment variables
#------------------------------------------------- 
ENV FSLDIR=/usr/local/fsl-6.0.3 
ENV PATH=${FSLDIR}/bin:$PATH
ENV FSLOUTPUTTYPE=NIFTI_GZ 
ENV YARN_VERSION 1.6.0


# Copy the current directory contents into the container
COPY . /computation

ADD server/. /server
WORKDIR /server
RUN npm i --production
CMD ["node", "/server/index.js"]
