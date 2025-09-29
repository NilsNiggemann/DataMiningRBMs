#!/bin/bash

PATHONPC="/scratch/shared"
LOCALPATH="./datasetRBMs"
SSH_HOSTNAME="pcictp"
MODE=""

while [[ $# -gt 0 ]]; do
	case "$1" in 
		-p|--put)
			MODE="PUT"
			shift
			;;
		-g|--get)
			MODE="GET"
			shift
			;;
		*)
			echo "Unknown flag. Usage: syncToPC -option. Valid options: --put to send local to remote and --get to fetch remote to local"
			;;
	esac
done

if [[ ${MODE} == "GET" ]]; then
	#pull from remote
	(set -x ; rsync -azuve ssh ${SSH_HOSTNAME}:${PATHONPC} ${LOCALPATH})
elif [[ ${MODE} == "PUT" ]]; then
	#push to remote
	(set -x ; rsync -azuve ssh ${LOCALPATH} ${SSH_HOSTNAME}:${PATHONPC})
fi
