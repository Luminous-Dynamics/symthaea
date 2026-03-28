#!/bin/sh
set -e

CONDUCTOR_CONFIG="${CONDUCTOR_CONFIG:-/etc/holochain/conductor-config.yaml}"
KEYSTORE_TYPE=$(grep 'type:' "$CONDUCTOR_CONFIG" | head -1 | awk '{print $2}')

echo "Mode: $KEYSTORE_TYPE"
echo "Config: $CONDUCTOR_CONFIG"

if [ "$KEYSTORE_TYPE" = "lair_server_in_proc" ]; then
    echo "Starting holochain with in-proc Lair..."
    echo "$LAIR_PASSPHRASE" | holochain --piped -c "$CONDUCTOR_CONFIG"
else
    LAIR_CONF=/data/holochain/lair-keystore-config.yaml
    if [ ! -f "$LAIR_CONF" ]; then
        echo "$LAIR_PASSPHRASE" | lair-keystore init --piped
    fi
    echo "$LAIR_PASSPHRASE" | lair-keystore server --piped &
    sleep 2
    LAIR_URL=$(grep '^connectionUrl:' "$LAIR_CONF" | cut -d' ' -f2)
    sed "s|connection_url:.*|connection_url: \"$LAIR_URL\"|" "$CONDUCTOR_CONFIG" > /tmp/conductor-config.yaml
    echo "$LAIR_PASSPHRASE" | holochain --piped -c /tmp/conductor-config.yaml
fi
