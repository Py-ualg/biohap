awk -F'\t' '
NR==2 {
  printf "#NAME"
  for (i=2; i<NF; i++) printf "\t%s", $i
  print ""
}
NR>2 && !/^#/ {
  for (i=1; i<NF; i++) printf "%s\t", $i
  print ""
}' ASV_table_original.txt > ASV_table.txt