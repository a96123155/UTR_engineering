#!/bin/bash
# Enhanced batch processing for RNA secondary structure visualization
# Usage: ./enhanced_rna_structures.sh input.fasta output_dir [options]

# Default options
FORMAT="pdf"
HIDE_NUCLEOTIDES=true
ENERGY_INFO=false
COLORS=false

# Check for required tools
for cmd in RNAfold RNAplot inkscape; do
	  if ! command -v $cmd &> /dev/null; then
		      echo "Error: $cmd is not installed. Please install it first."
		          if [ "$cmd" = "RNAfold" ] || [ "$cmd" = "RNAplot" ]; then
				        echo "Install ViennaRNA package: sudo apt install viennarna"
					    elif [ "$cmd" = "inkscape" ]; then
						          echo "Install Inkscape: sudo apt install inkscape"
							      fi
							          exit 1
								    fi
							    done

							    # Process command line options
							    while getopts ":f:s:e:c:" opt; do
								      case $opt in
									          f) FORMAT="$OPTARG" ;;     # Format: pdf, png, svg
										      s) HIDE_NUCLEOTIDES="$OPTARG" ;;  # true/false to hide nucleotides
										          e) ENERGY_INFO="$OPTARG" ;;       # true/false to show energy information
											      c) COLORS="$OPTARG" ;;            # true/false for colored structure
											          \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
												    esac
											    done
											    shift $((OPTIND - 1))

											    if [ "$#" -lt 2 ]; then
												      echo "Usage: $0 <input_fasta> <output_directory> [-f format] [-s hide_nucleotides] [-e energy_info] [-c colors]"
												        echo "Options:"
													  echo "  -f format: pdf, png, or svg (default: pdf)"
													    echo "  -s hide_nucleotides: true or false (default: true)"
													      echo "  -e energy_info: true or false (default: false)"
													        echo "  -c colors: true or false (default: false)"
														  exit 1
											    fi

											    INPUT_FILE="$1"
											    OUTPUT_DIR="$2"

											    # Validate input file
											    if [ ! -f "$INPUT_FILE" ]; then
												      echo "Error: Input file '$INPUT_FILE' not found"
												        exit 1
											    fi

											    mkdir -p "$OUTPUT_DIR"
											    TEMP_DIR=$(mktemp -d)
											    echo "Working in temporary directory: $TEMP_DIR"

											    # Configure RNAplot options
											    RNAPLOT_OPTS="-o svg"
											    if [ "$HIDE_NUCLEOTIDES" = true ]; then
												      RNAPLOT_OPTS="$RNAPLOT_OPTS -t 0"
											    fi

											    if [ "$COLORS" = true ]; then
												      RNAPLOT_OPTS="$RNAPLOT_OPTS -c"
											    fi

											    # Process FASTA file with potential multi-line sequences
											    echo "Processing sequences from $INPUT_FILE..."

											    # Split FASTA into individual sequence files in temp directory
											    awk '
											    BEGIN {n=0; out=""} 
											    /^>/ {if(out!="") {close(out)}; n++; out=sprintf("'"$TEMP_DIR"'/seq_%d.fa", n); print > out; next} 
											    {print >> out}
											    ' "$INPUT_FILE"

											    echo "Processing $(ls "$TEMP_DIR"/seq_*.fa | wc -l) sequences..."

											    # Process each sequence file
											    for SEQ_FILE in "$TEMP_DIR"/seq_*.fa; do
												      # Get sequence name from FASTA header
												        SEQ_NAME=$(head -n 1 "$SEQ_FILE" | sed 's/^>//' | sed 's/[^a-zA-Z0-9_-]/_/g')
													  echo "Processing: $SEQ_NAME"
													    
													    # Extract sequence (handling multi-line sequences)
													      grep -v "^>" "$SEQ_FILE" | tr -d '\n\r ' > "$TEMP_DIR/${SEQ_NAME}.seq"
													        
													        # Check if sequence is empty
														  if [ ! -s "$TEMP_DIR/${SEQ_NAME}.seq" ]; then
															      echo "Warning: Empty sequence for $SEQ_NAME, skipping"
															          continue
																    fi
																      
																      # Process with RNAfold
																        cd "$TEMP_DIR"
																	  echo "Running RNAfold..."
																	    if [ "$ENERGY_INFO" = true ]; then
																		        cat "${SEQ_NAME}.seq" | RNAfold > "${SEQ_NAME}_fold.txt"
																			  else
																				      cat "${SEQ_NAME}.seq" | RNAfold --noPS > "${SEQ_NAME}_fold.txt"
																				        fi
																					  
																					  # Check if RNAfold was successful
																					    if [ ! -f "${SEQ_NAME}_fold.txt" ]; then
																						        echo "Error: RNAfold failed for $SEQ_NAME"
																							    cd - > /dev/null
																							        continue
																								  fi
																								    
																								    # Extract the structure
																								      STRUCTURE=$(tail -n 1 "${SEQ_NAME}_fold.txt" | awk '{print $1}')
																								        
																								        # Check if structure was generated
																									  if [ -z "$STRUCTURE" ]; then
																										      echo "Error: No structure generated for $SEQ_NAME"
																										          cd - > /dev/null
																											      continue
																											        fi
																												  
																												  echo "Structure notation obtained. Length: ${#STRUCTURE} bases"
																												    
																												    # Prepare input for RNAplot
																												      echo "$(cat ${SEQ_NAME}.seq) $STRUCTURE" > "${SEQ_NAME}_plot_input.txt"
																												        
																												        # Generate the plot
																													  echo "Generating structure plot with options: $RNAPLOT_OPTS"
																													    cat "${SEQ_NAME}_plot_input.txt" | RNAplot $RNAPLOT_OPTS
																													      
																													      # Find the generated SVG
																													        SVG_FILE=$(find . -maxdepth 1 -name "*.svg" | tail -n 1)
																														  
																														  if [ -z "$SVG_FILE" ] || [ ! -f "$SVG_FILE" ]; then
																															      echo "Error: No SVG file generated for $SEQ_NAME"
																															          cd - > /dev/null
																																      continue
																																        fi
																																	  
																																	  echo "SVG file generated: $SVG_FILE"
																																	    
																																	    # Convert to desired format
																																	      case "$FORMAT" in
																																		          pdf)
																																				        echo "Converting to PDF..."
																																					      inkscape --export-filename="${SEQ_NAME}.pdf" "$SVG_FILE"
																																					            if [ -f "${SEQ_NAME}.pdf" ]; then
																																							            cp "${SEQ_NAME}.pdf" "$OUTPUT_DIR/"
																																								            echo "PDF saved to $OUTPUT_DIR/${SEQ_NAME}.pdf"
																																									          else
																																											          echo "Error: Failed to create PDF"
																																												        fi
																																													      ;;
																																													          png)
																																															        echo "Converting to PNG..."
																																																      inkscape --export-filename="${SEQ_NAME}.png" --export-dpi=300 "$SVG_FILE"
																																																            if [ -f "${SEQ_NAME}.png" ]; then
																																																		            cp "${SEQ_NAME}.png" "$OUTPUT_DIR/"
																																																			            echo "PNG saved to $OUTPUT_DIR/${SEQ_NAME}.png"
																																																				          else
																																																						          echo "Error: Failed to create PNG"
																																																							        fi
																																																								      ;;
																																																								          svg)
																																																										        echo "Saving SVG file..."
																																																											      cp "$SVG_FILE" "$OUTPUT_DIR/${SEQ_NAME}.svg"
																																																											            echo "SVG saved to $OUTPUT_DIR/${SEQ_NAME}.svg"
																																																												          ;;
																																																													    esac
																																																													      
																																																													      cd - > /dev/null
																																																												      done

																																																												      # Clean up
																																																												      echo "Cleaning up temporary files..."
																																																												      rm -rf "$TEMP_DIR"

																																																												      echo "Completed: All RNA structures have been processed and saved to $OUTPUT_DIR"
