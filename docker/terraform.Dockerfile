FROM hashicorp/terraform:1.14
RUN apk add --no-cache aws-cli
WORKDIR /work
ENTRYPOINT [""]