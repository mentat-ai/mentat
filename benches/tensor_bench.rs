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

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mentat::tensor::{DataType, Tensor};

fn bench_tensor_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tensor Add");
    for size in [128, 512, 1024].iter() {
        group.bench_with_input(criterion::BenchmarkId::from_parameter(size), size, |b, &s| {
            let mut t1 = Tensor::new(vec![s, s], DataType::Float32).unwrap();
            let mut t2 = Tensor::new(vec![s, s], DataType::Float32).unwrap();
            
            // Fill with some dummy data
            t1.data.fill(1.0);
            t2.data.fill(2.0);

            b.iter(|| {
                let _ = black_box(&t1).add(black_box(&t2));
            });
        });
    }
    group.finish();
}

fn bench_tensor_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tensor MatMul");
    for size in [128, 512].iter() {
        group.bench_with_input(criterion::BenchmarkId::from_parameter(size), size, |b, &s| {
            let mut t1 = Tensor::new(vec![s, s], DataType::Float32).unwrap();
            let mut t2 = Tensor::new(vec![s, s], DataType::Float32).unwrap();
            
            t1.data.fill(1.0);
            t2.data.fill(2.0);

            b.iter(|| {
                let _ = black_box(&t1).matmul(black_box(&t2));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tensor_add, bench_tensor_matmul);
criterion_main!(benches);
