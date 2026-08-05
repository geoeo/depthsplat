#!/usr/bin/env bash

SCRIPTPATH="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"

user_group_id=$(id -g)
user_group=$(id -gn)

id | cut -d = -f 4 | sed -E "s/$user_group_id\($user_group\),?//" | sed -E "s/,+$//" > .devcontainer/host_groups