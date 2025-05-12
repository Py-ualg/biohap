if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <ASV_table.tsv>"
  exit 1
fi

input_file="$1"

awk -F'\t' '
NR==2 {
  printf "#NAME"
  for (i=2; i<NF; i++) printf "\t%s", $i
  print ""
}
NR>2 && !/^#/ {
  printf "%s", $1  # print the first column as-is
  for (i=2; i<NF; i++) {
          printf "\t%s", int($i)
  }
  print ""
}' "$input_file" > ASV_table_MA.txt