#  Author: Rupak Raj (rughimire)
#  Affiliation: NAIST
#  Date: 2026.02

download_path=${PWD}
echo "Open SLR Dataset for Nepali: SLR54 (for details run with --help)"
function print_help_message {
  echo "usage:
  $0 \\ (The arguments in [] are optional while other arguments must be given by your run.sh.)
    --download_path DOWNLOAD_PATH \\       # The path to place the downloaded dataset. (default: \$PWD)" >&2
  exit 1
}

### get args from the command line ###
while getopts ":h-:" optchar; do
  case "${optchar}" in
    -)
      case "${OPTARG}" in
        download_path)
          val="${!OPTIND}"; OPTIND=$(( OPTIND + 1 ))
          download_path=${val}
          ;;
        help)
          print_help_message
          ;;
        ?)
          echo "Unknown variable --$OPTARG"
          exit 1 ;;
      esac
      ;;
    h)
      print_help_message
      ;;
    *)
      echo "Please refer to an argument by '--'."
      exit 1 ;;
  esac
done

# data dump path
mkdir -p ${download_path}/data/wav

echo "No download yet!!! :-("
