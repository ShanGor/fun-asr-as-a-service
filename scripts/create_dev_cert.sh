#!/usr/bin/env bash
# Create a short-lived self-signed certificate for local/remote testing.
#
# Usage:
#   scripts/create_dev_cert.sh [hostname-or-ip ...]
#
# With no arguments, the certificate covers localhost, this machine's short
# hostname, and every IPv4 address currently reported by `hostname -I`.
# Set FORCE=1 to replace an existing certificate/key pair.
set -euo pipefail

cd "$(dirname "$0")/.."

out_dir="${FUNASR_TLS_DIR:-deploy/tls}"
cert_file="${FUNASR_TLS_CERT:-$out_dir/fullchain.pem}"
key_file="${FUNASR_TLS_KEY:-$out_dir/privkey.pem}"

if [[ "${FORCE:-0}" != "1" ]] && { [[ -e "$cert_file" ]] || [[ -e "$key_file" ]]; }; then
  echo "Refusing to overwrite existing TLS files. Set FORCE=1 to replace them." >&2
  exit 1
fi

mkdir -p "$out_dir"
chmod 700 "$out_dir"

names=("$@")
if ((${#names[@]} == 0)); then
  names=(localhost "$(hostname -s)")
  while read -r address; do
    [[ -n "$address" ]] && names+=("$address")
  done < <(hostname -I 2>/dev/null | tr ' ' '\n')
fi

san_entries=()
for name in "${names[@]}"; do
  [[ -z "$name" ]] && continue
  if [[ "$name" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    san_entries+=("IP:$name")
  else
    san_entries+=("DNS:$name")
  fi
done

if ((${#san_entries[@]} == 0)); then
  echo "No hostnames or IP addresses were provided." >&2
  exit 1
fi

san="$(IFS=,; echo "${san_entries[*]}")"
common_name="${names[0]}"

openssl req \
  -x509 \
  -newkey rsa:2048 \
  -sha256 \
  -nodes \
  -days "${FUNASR_TLS_DAYS:-30}" \
  -subj "/CN=$common_name" \
  -addext "subjectAltName=$san" \
  -keyout "$key_file" \
  -out "$cert_file" \
  >/dev/null 2>&1

chmod 600 "$key_file"
chmod 644 "$cert_file"

echo "Created certificate: $cert_file"
echo "Created private key: $key_file"
openssl x509 -in "$cert_file" -noout -subject -issuer -dates -ext subjectAltName
