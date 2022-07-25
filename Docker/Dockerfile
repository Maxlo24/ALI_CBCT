FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update && \
	apt-get install -y git \
	# apt-get install -y nvidia-container-toolkit \
	# libx11-6 \
	# libgl1 \
	# libopengl0 \
	# libegl1 \
	wget\
	-y unzip


RUN mkdir /app
RUN mkdir data
WORKDIR /app/data

# ARG MODEL_VERSION="1.0.2-beta.1"
# RUN wget https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v$MODEL_VERSION/ALL_MODELS.zip

# RUN unzip ALL_MODELS.zip
# RUN rm -rf ALL_MODELS.zip


ARG RELEASE_VERSION="1.0.0-beta.1"

WORKDIR /app
RUN wget https://github.com/Maxlo24/ALI_CBCT/archive/refs/tags/v$RELEASE_VERSION.zip
RUN unzip v$RELEASE_VERSION.zip
# RUN unzip AMASSS_CBCT-$RELEASE.zip
# RUN unzip AMASSS_CBCT-1.0.0-alpha.zip

RUN pip install -r ALI_CBCT-$RELEASE_VERSION/requirements.txt

RUN mv ALI_CBCT-$RELEASE_VERSION/src/AGENT ALI_CBCT
# RUN mv /app/$RELEASE_VERSION/Data /app/Data
RUN rm -rf v$RELEASE_VERSION.zip \
	rm -rf v$RELEASE_VERSION



RUN mkdir scans
RUN mkdir models
