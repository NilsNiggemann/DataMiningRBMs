#!/bin/bash

PATHONPC="/scratch/shared/"
LOCALPATH="./data"

# SSH_HOSTNAME=$RBM_DATA_HOST
SSH_HOSTNAME=ICTP_office
MODE=""
FILES=()

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
		-*)
			echo "Unknown flag: $1"
			echo "Usage: $0 [--put|--get] [folder1 folder2 ...]"
			exit 1
			;;
		*)
			FILES+=("$1")
			shift
			;;
	esac
done

if [[ -z ${MODE} ]]; then
	echo "Error: Must specify --put or --get"
	echo "Usage: $0 [--put|--get] [folder1 folder2 ...]"
	exit 1
fi

# When no folders/files are named explicitly, sync every top-level entry
# under the source directory as its own parallel rsync job, instead of one
# single rsync over the whole tree (a single rsync only ever uses one
# connection, so it can't use more than one stream of bandwidth at a time).
if [[ ${MODE} == "GET" ]]; then
	if [[ ${#FILES[@]} -eq 0 ]]; then
		mapfile -t FILES < <(ssh ${SSH_HOSTNAME} "ls ${PATHONPC}")
	fi
	for f in "${FILES[@]}"; do
		(set -x ; rsync -Lazuve ssh ${SSH_HOSTNAME}:${PATHONPC}/${f} ${LOCALPATH}/) &
	done
	wait
elif [[ ${MODE} == "PUT" ]]; then
	if [[ ${#FILES[@]} -eq 0 ]]; then
		mapfile -t FILES < <(ls "${LOCALPATH}")
	fi
	for f in "${FILES[@]}"; do
		(set -x ; rsync -Lazuve ssh ${LOCALPATH}/${f} ${SSH_HOSTNAME}:${PATHONPC}/) &
	done
	wait
fi
