import numpy as np

def einsum(subscripts, *operands):
    input_subscripts, output_subscript = subscripts.split('->')
    input_subscripts = input_subscripts.split(',')

    all_indices = ''.join(input_subscripts)
    
    dim_sizes = {}
    for s, op in zip(input_subscripts, operands):
        for idx, dim in zip(s, op.shape):
            if idx in dim_sizes and dim_sizes[idx] != dim:
                raise ValueError(f"차원 불일치: 인덱스 {idx}")
            dim_sizes[idx] = dim
    
    # 메시그리드 생성 (모든 인덱스 조합)
    mesh = np.ix_(*[range(dim_sizes[i]) for i in all_indices])
    
    # 텐서 곱 계산 (브로드캐스팅 활용)
    product = np.ones(mesh[0].shape)
    for s, op in zip(input_subscripts, operands):
        idx = tuple(mesh[all_indices.index(i)] for i in s)
        product = product * op[idx]  # 주의: *= 대신 = 사용
    
    # 출력에 없는 인덱스 축으로 합산
    sum_indices = [i for i in all_indices if i not in output_subscript]
    for idx in sum_indices:
        axis = all_indices.index(idx)
        product = product.sum(axis=axis, keepdims=True)
    
    # 불필요한 차원 제거 (예: (2,1) → (2,))
    for i in reversed(range(len(product.shape))):
        if product.shape[i] == 1:
            product = np.squeeze(product, axis=i)
    
    return product

# 사용 예시
A = np.array([[1, 2, 3], [4, 5, 6]])      # 2x3
B = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
result = einsum('ik,ki->ik', A, B)
print(A.shape)
print(B.shape)
print(result)