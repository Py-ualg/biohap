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
}' ASV_table.tsv > ASV_table_MA.txt