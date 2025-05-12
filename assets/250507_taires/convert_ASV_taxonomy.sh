if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <ASV_table.tsv>"
  exit 1
fi

input_file="$1"

# Remove trailing characters (spaces or tabs) and convert Windows-style line endings to Unix-style
sed -e 's/[ \t]*$//' -e 's/\r$//' "$input_file" > ASV_table_cleaned.txt

awk -F'\t' '
BEGIN {
  OFS = "\t"
  print "#TAXONOMY", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"
}
NR > 1 && !/^#/ {
  # Reset taxonomy output array
  for (i = 1; i <= 7; i++) tax[i] = ""

  # Split taxonomy string
  split($NF, rawtax, "; ")

  # Clean taxonomy ranks and store them
  for (i = 1; i <= length(rawtax); i++) {
    gsub(/^.*__/, "", rawtax[i])
    tax[i] = rawtax[i]
  }

  # Print OTU ID and cleaned taxonomy ranks
  print $1, tax[1], tax[2], tax[3], tax[4], tax[5], tax[6], tax[7]
}
' ASV_table_cleaned.txt > taxonomy_dp_tabs.tsv

sed '/^[ \t]*$/d' taxonomy_dp_tabs.tsv > taxonomy_MA.txt
rm taxonomy_dp_tabs.tsv
rm ASV_table_cleaned.txt