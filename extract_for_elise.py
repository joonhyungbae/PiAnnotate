#!/usr/bin/env python3
"""
Extract for_elise_data.tar.xz to for_elise directory
"""

import os
import sys
import tarfile
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def extract_for_elise():
    """Extract for_elise_data.tar.xz to for_elise directory"""
    
    # Get script directory and project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir
    
    # Paths
    archive_file = project_root / "for_elise_data.tar.xz"
    output_dir = project_root / "for_elise"
    
    # Check if archive file exists
    if not archive_file.exists():
        print(f"❌ Archive file not found: {archive_file}")
        print(f"   Please make sure 'for_elise_data.tar.xz' is in the project root.")
        return 1
    
    # Check file size
    file_size_gb = archive_file.stat().st_size / (1024 ** 3)
    print(f"📦 Found archive: {archive_file.name}")
    print(f"   Size: {file_size_gb:.2f} GB")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if output directory is empty
    if any(output_dir.iterdir()):
        response = input(f"⚠️  Output directory '{output_dir.name}' is not empty. Continue? (y/N): ")
        if response.lower() != 'y':
            print("   Extraction cancelled.")
            return 1
    
    # Extract archive
    print(f"📂 Extracting to {output_dir}...")
    print("   This may take a while (extracting 42GB file)...")
    print()
    
    try:
        # First, check the archive structure by reading the first member
        # We need to detect if the archive has a 'for_elise' root directory
        strip_prefix = None
        with tarfile.open(archive_file, 'r:xz') as tar_check:
            first_member = tar_check.next()
            if first_member is None:
                print("❌ Archive is empty")
                return 1
            
            # Detect root directory name
            root_dir = first_member.name.split('/')[0] if '/' in first_member.name else first_member.name
            
            # If archive contains 'for_elise' directory, strip it to avoid nested directories
            if root_dir == "for_elise":
                strip_prefix = "for_elise/"
                print(f"   Detected root directory: {root_dir}")
                print(f"   Stripping prefix to avoid nested directories...")
            else:
                print(f"   Archive root: {root_dir}")
        
        # Now extract with the detected prefix
        with tarfile.open(archive_file, 'r:xz') as tar:
            file_count = 0
            
            if HAS_TQDM:
                # Estimate total size for progress bar (use file size as approximation)
                total_size = archive_file.stat().st_size
                pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Extracting")
                
                # Extract members one by one without loading all into memory
                member = tar.next()
                while member is not None:
                    # Strip the root directory prefix if needed
                    if strip_prefix and member.name.startswith(strip_prefix):
                        # Adjust the member's path
                        original_name = member.name
                        new_name = member.name[len(strip_prefix):]
                        if new_name:  # Skip if empty (root directory itself)
                            member.name = new_name
                            tar.extract(member, output_dir)
                            member.name = original_name  # Restore for progress tracking
                            file_count += 1
                            # Update progress based on extracted file size
                            if member.size > 0:
                                pbar.update(member.size)
                            else:
                                # For directories or small files, update by a small amount
                                pbar.update(1024)
                    elif not strip_prefix:
                        # No prefix to strip, extract as is
                        tar.extract(member, output_dir)
                        file_count += 1
                        if member.size > 0:
                            pbar.update(member.size)
                        else:
                            pbar.update(1024)
                    
                    member = tar.next()
                
                pbar.close()
            else:
                # Fallback: extract without progress bar (but still streaming)
                print("   Extracting files...")
                member = tar.next()
                while member is not None:
                    # Strip the root directory prefix if needed
                    if strip_prefix and member.name.startswith(strip_prefix):
                        original_name = member.name
                        new_name = member.name[len(strip_prefix):]
                        if new_name:
                            member.name = new_name
                            tar.extract(member, output_dir)
                            member.name = original_name  # Restore
                            file_count += 1
                    elif not strip_prefix:
                        tar.extract(member, output_dir)
                        file_count += 1
                    
                    if file_count % 1000 == 0:
                        print(f"   Extracted {file_count} files...", end='\r')
                    member = tar.next()
                print(f"   Extracted {file_count} files")
        
        print()
        print(f"✅ Extraction complete! {file_count} files extracted to {output_dir}")
        return 0
        
    except KeyboardInterrupt:
        print()
        print("⚠️  Extraction interrupted by user")
        return 1
    except tarfile.TarError as e:
        print(f"❌ Extraction failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(extract_for_elise())

