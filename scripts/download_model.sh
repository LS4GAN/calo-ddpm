#!/usr/bin/env bash

OUTDIR="${JETGEN_OUTDIR:-outdir}"
ZENODO_BASE="https://zenodo.org/records/12535659"

# NOTE: regen command
# $ sha256sum *.zip | awk '{printf "[%s]=%s\n", $2, $1}'

declare -A CHECKSUMS=(
	[cent0_ddpm_seed0.zip]=6075d1321b8b73ea1d5b868adcf76fad6337310a8b4ffe7bb008fd1a14d72a24
	[cent0_ddpm_seed1.zip]=8e4a76f99b2423bcc5a2d0b5b79c48e47bd702446140b8823f8b442bb7348c42
	[cent0_ddpm_seed2.zip]=ec6231a8577aadf0bc181b7ed450aedbe3b9dcb1202088b715f46a9bc3f57997
	[cent0_ddpm_seed3.zip]=bb4fa37f0bca339426a29f5148b4eb29839c583104c5b36eeaa15c238f71650e
	[cent0_ddpm_seed4.zip]=e707bde59cf7084c50de73b53aedcccad3700b7a194777860899c14cae281091
	[cent0_gan_seed0.zip]=3d220c8941dd6a506a5e3e5f5b06855d634b060a58da02b740da22da49af1c7d
	[cent0_gan_seed1.zip]=5784f932cf31888d56afd6d216f4b28c35400fbef7722741ddfbb686e581f66d
	[cent0_gan_seed2.zip]=a3d2d8dbd6a047102ec73123d33a7db2adeeb50a1ee4dbcaade72ed095158d59
	[cent0_gan_seed3.zip]=0c809356f00668897f2133f343093b5a358488d7bf6cee48e245e8073095adec
	[cent0_gan_seed4.zip]=72c8fee51c17c7bdaa1f7bfe08b37ada2625ce02ff7713a68d1a717f84f7f65f
	[cent4_ddpm_seed0.zip]=fad293d7e3ea3ed58af63e04d51befb88eaf1e6b398397d4028976df0777de5c
	[cent4_ddpm_seed1.zip]=5a6e635ca537f8fb6857b52cfabc87d8566b5f78358fd61806285774c3bd290d
	[cent4_ddpm_seed2.zip]=a50c209a00514644fa8de7f9efb624b27007ff0dee37121696ca6ea2b4dbf568
	[cent4_ddpm_seed3.zip]=acc000c45a707572de10468f6030e08267b938eca953bb828cd2a71f362c911c
	[cent4_ddpm_seed4.zip]=dbd0c35d57f1bb8f2947098a0714cefcdb1f225318da29e566dd983e58930f83
	[cent4_gan_seed0.zip]=6bc264866bfa67a974208c37559f5048f96bfb3674a58237c7c26e14c30c3739
	[cent4_gan_seed1.zip]=ca801913decacad826678197c1d30329456e113dfc22809cb9b6435f965503cb
	[cent4_gan_seed2.zip]=e22e8f019390067d6efbac6956c82b6a26d17b325b2546c9d7af54e74939b997
	[cent4_gan_seed3.zip]=5613a17f5f067f773b3578b821bf0635947aec8de58090e9b5769681d1f102b8
	[cent4_gan_seed4.zip]=22a11cca7cded4abdf8203e8e17f87a270606f9f423abb8450d557953db28c72
)

die ()
{
    echo "${*}"
    exit 1
}

usage ()
{
    cat <<EOF
USAGE: download_model.sh [-h|--help] MODEL

Download and extract a pre-trained model. MODEL is one of
EOF
    printf "  - %s\n" "${!CHECKSUMS[@]}" | sort

    if [[ $# -gt 0 ]]
    then
        die "${*}"
    else
        exit 0
    fi
}

exec_or_die ()
{
    "${@}" || die "Failed to execute: '${*}'"
}

get_file_name ()
{
    local dataset="${1}"
    local full="${2}"

    if [[ "${full}" == 1 ]]
    then
        echo "${dataset}_full.zip"
    else
        echo "${dataset}_only_gen.zip"
    fi
}

calc_sha256_hash ()
{
    local path="${1}"
    sha256sum "${path}" | cut -d ' ' -f 1 | tr -d '\n'
}

download_zenodo_file ()
{
    local archive="${1}"
    local dest="${2}"

    local url="${ZENODO_BASE}/files/${archive}"

    exec_or_die wget "${url}" --output-document "${dest}"
}

download_archive ()
{
    local archive="${1}"
    local save_path="${2}"

    if [[ ! -e "${save_path}" ]]
    then
        exec_or_die mkdir -p "$(dirname "${save_path}")"
        download_zenodo_file "${archive}" "${save_path}"
    fi

    local null_csum="${CHECKSUMS[${archive}]}"

    if [[ -n "${null_csum}" ]]
    then
        # shellcheck disable=SC2155
        local test_csum="$(calc_sha256_hash "${save_path}")"

        if [[ "${test_csum}" == "${null_csum}" ]]
        then
            echo " - Checksum valid"
        else
            die "Checksum mismatch for '${save_path}' "\
                "${test_csum} vs ${null_csum}"
        fi
    fi
}

download_and_extract_zip ()
{
    local archive="${1}"
    local save_path="${2}"
    local extract_root="${3}"

    download_archive  "${archive}" "${save_path}"
    exec_or_die unzip "${save_path}" -d "${extract_root}"

    echo " - Model is downloaded to: '${save_path}'"
}

download_model ()
{
    local archive="${1}"

    local save_path="${OUTDIR}/${archive}"
    local extract_root="${OUTDIR}/calo-ddpm"

    download_and_extract_zip "${archive}" "${save_path}" "${extract_root}"
}

MODEL=

while [ $# -gt 0 ]
do
    case "$1" in
        -h|--help|help)
            usage
            ;;
        *)
            [[ -z "${CHECKSUMS[$1]}" ]] && usage "Unknown model '$1'"
            [[ -n "${MODEL}" ]] && usage "Model is already specified"

            MODEL="${1}"
            shift
            ;;
    esac
done

[[ -z "${MODEL}" ]] && usage
download_model "${MODEL}" "${FULL}"

