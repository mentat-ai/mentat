// Copyright 2026 Mentat AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::tensor::{DataType, Tensor};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;

/// Loader handles reading model weights from Safetensors files.
pub struct Loader;

impl Loader {
    /// Loads weights from a .safetensors file and returns a mapping of name -> Tensor.
    pub fn load_safetensors<P: AsRef<Path>>(
        path: P,
    ) -> Result<std::collections::HashMap<String, Tensor>, String> {
        let file = File::open(path).map_err(|e| format!("failed to open weight file: {}", e))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("failed to mmap file: {}", e))? };

        // We need to keep the mmap alive as long as the SafeTensors struct exists
        // However, for our simple Tensor implementation which copies data into a Vec,
        // we can just parse it and discard the mmap after copying.
        let st =
            SafeTensors::deserialize(&mmap).map_err(|e| format!("safetensors error: {}", e))?;

        let mut weights = std::collections::HashMap::new();

        for (name, view) in st.tensors() {
            let shape = view.shape().to_vec();
            
            // Convert bytes to f32. This assumes Little Endian.
            // Modern models often use BF16 or F16. We upcast them to F32 for CPU processing.
            let bytes = view.data();
            let mut data = Vec::with_capacity(bytes.len() / 2);

            match view.dtype() {
                safetensors::Dtype::F32 => {
                    for chunk in bytes.chunks_exact(4) {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(chunk);
                        data.push(f32::from_le_bytes(b));
                    }
                }
                safetensors::Dtype::BF16 => {
                    for chunk in bytes.chunks_exact(2) {
                        let mut b = [0u8; 2];
                        b.copy_from_slice(chunk);
                        let u_val = u16::from_le_bytes(b);
                        // BF16 to F32: just shift left by 16
                        let f_val = f32::from_bits((u_val as u32) << 16);
                        data.push(f_val);
                    }
                }
                safetensors::Dtype::F16 => {
                    for chunk in bytes.chunks_exact(2) {
                        let mut b = [0u8; 2];
                        b.copy_from_slice(chunk);
                        let u_val = u16::from_le_bytes(b);
                        
                        // F16 to F32 conversion
                        let sign = (u_val & 0x8000) >> 15;
                        let exp = (u_val & 0x7C00) >> 10;
                        let mut mant = u_val & 0x03FF;

                        let f_val = if exp == 0 {
                            if mant == 0 {
                                f32::from_bits((sign as u32) << 31)
                            } else {
                                // Subnormal
                                f32::from_bits(((sign as u32) << 31) | (127 - 14) << 23) * (mant as f32 / 1024.0)
                            }
                        } else if exp == 31 {
                            if mant == 0 {
                                if sign == 0 { f32::INFINITY } else { f32::NEG_INFINITY }
                            } else {
                                f32::NAN
                            }
                        } else {
                            f32::from_bits(((sign as u32) << 31) | ((exp as u32 + 127 - 15) << 23) | ((mant as u32) << 13))
                        };
                        data.push(f_val);
                    }
                }
                _ => {
                    return Err(format!(
                        "unsupported data type in tensor {}: {:?}",
                        name,
                        view.dtype()
                    ))
                }
            }

            let tensor = Tensor {
                shape,
                dtype: DataType::Float32,
                data,
            };
            weights.insert(name.to_string(), tensor);
        }

        Ok(weights)
    }
}
