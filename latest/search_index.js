var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Tensors-1",
    "page": "Home",
    "title": "Tensors",
    "category": "section",
    "text": "Efficient computations with symmetric and non-symmetric tensors in Julia."
},

{
    "location": "index.html#Introduction-1",
    "page": "Home",
    "title": "Introduction",
    "category": "section",
    "text": "This Julia package provides fast operations with symmetric and non-symmetric tensors of order 1, 2 and 4. The Tensors are allocated on the stack which means that there is no need to preallocate output results for performance. Unicode infix operators are provided such that the tensor expression in the source code is similar to the one written with mathematical notation. When possible, symmetry of tensors is exploited for better performance. Supports Automatic Differentiation to easily compute first and second order derivatives of tensorial functions."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "Tensors is a registered package and so can be installed viaPkg.add(\"Tensors\")"
},

{
    "location": "index.html#Manual-Outline-1",
    "page": "Home",
    "title": "Manual Outline",
    "category": "section",
    "text": "Pages = [\n    \"man/constructing_tensors.md\",\n    \"man/indexing.md\",\n    \"man/binary_operators.md\",\n    \"man/other_operators.md\",\n    \"man/storing_tensors.md\",\n]\nDepth = 1"
},

{
    "location": "index.html#Demos-1",
    "page": "Home",
    "title": "Demos",
    "category": "section",
    "text": "Pages = [\n    \"demos.md\"]\nDepth = 1"
},

{
    "location": "man/constructing_tensors.html#",
    "page": "Constructing tensors",
    "title": "Constructing tensors",
    "category": "page",
    "text": "DocTestSetup = quote\n    srand(1234)\n    using Tensors\nend"
},

{
    "location": "man/constructing_tensors.html#Constructing-tensors-1",
    "page": "Constructing tensors",
    "title": "Constructing tensors",
    "category": "section",
    "text": "Tensors can be created in multiple ways but they usually include running a function on tensor types of which there are two kinds, Tensor{order, dim, T} for non-symmetric tensors and SymmetricTensor{order, dim, T} for symmetric tensors. The parameter order is an integer of value 1, 2 or 4, excluding 1 for symmetric tensors. The second parameter dim is an integer which corresponds to the dimension of the tensor and can be 1, 2 or 3. The last parameter T is the number type that the tensors contain, i.e. Float64 or Float32."
},

{
    "location": "man/constructing_tensors.html#zero_tensors-1",
    "page": "Constructing tensors",
    "title": "Zero tensors",
    "category": "section",
    "text": "A tensor with only zeros is created using the function zero, applied to the type of tensor that should be created:julia> zero(Tensor{1, 2})\n2-element Tensors.Tensor{1,2,Float64,2}:\n 0.0\n 0.0By default, a tensor of Float64s is created, but by explicitly giving the T parameter this can be changed:julia> zero(SymmetricTensor{4, 2, Float32})\n2×2×2×2 Tensors.SymmetricTensor{4,2,Float32,9}:\n[:, :, 1, 1] =\n 0.0  0.0\n 0.0  0.0\n\n[:, :, 2, 1] =\n 0.0  0.0\n 0.0  0.0\n\n[:, :, 1, 2] =\n 0.0  0.0\n 0.0  0.0\n\n[:, :, 2, 2] =\n 0.0  0.0\n 0.0  0.0A Julia Array with zeroed tensors can be created with zeroes, with the tensor type and dimensions of the array as arguments:julia> zeros(Tensor{2,2}, 2, 3)\n2×3 Array{Tensors.Tensor{2,2,Float64,4},2}:\n [0.0 0.0; 0.0 0.0]  [0.0 0.0; 0.0 0.0]  [0.0 0.0; 0.0 0.0]\n [0.0 0.0; 0.0 0.0]  [0.0 0.0; 0.0 0.0]  [0.0 0.0; 0.0 0.0]"
},

{
    "location": "man/constructing_tensors.html#Constant-tensors-1",
    "page": "Constructing tensors",
    "title": "Constant tensors",
    "category": "section",
    "text": "A tensor filled with ones is created using the function ones, applied to the type of tensor that should be created:julia> ones(Tensor{2,2})\n2×2 Tensors.Tensor{2,2,Float64,4}:\n 1.0  1.0\n 1.0  1.0By default, a tensor of Float64s is created, but by explicitly giving the T parameter this can be changed, like for zero.note: Note\nThe function ones has double meaning: it can create a tensor filled with ones (as described above) or create a Julia Array with identity tensors. Thus, to create an Array with tensors filled with ones, instead use array comprehension:julia> [ones(Tensor{2,2}) for i in 1:2, j in 1:3]\n2×3 Array{Tensors.Tensor{2,2,Float64,4},2}:\n [1.0 1.0; 1.0 1.0]  [1.0 1.0; 1.0 1.0]  [1.0 1.0; 1.0 1.0]\n [1.0 1.0; 1.0 1.0]  [1.0 1.0; 1.0 1.0]  [1.0 1.0; 1.0 1.0]"
},

{
    "location": "man/constructing_tensors.html#Random-tensors-1",
    "page": "Constructing tensors",
    "title": "Random tensors",
    "category": "section",
    "text": "A tensor with random numbers is created using the function rand, applied to the type of tensor that should be created:julia> rand(Tensor{2, 3})\n3×3 Tensors.Tensor{2,3,Float64,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837By specifying the type, T, a tensor of different type can be obtained:julia> rand(SymmetricTensor{2,3,Float32})\n3×3 Tensors.SymmetricTensor{2,3,Float32,6}:\n 0.0107703  0.305865  0.2082\n 0.305865   0.405684  0.257278\n 0.2082     0.257278  0.958491"
},

{
    "location": "man/constructing_tensors.html#identity_tensors-1",
    "page": "Constructing tensors",
    "title": "Identity tensors",
    "category": "section",
    "text": "Identity tensors can be created for orders 2 and 4. The components of the second order identity tensor mathbfI are defined as I_ij = delta_ij, where delta_ij is the Kronecker delta. The fourth order identity tensor mathsfI is the resulting tensor from taking the derivative of a second order tensor mathbfA with itself:mathsfI = fracpartial mathbfApartial mathbfA Leftrightarrow I_ijkl = fracpartial A_ijpartial A_kl = delta_ik delta_jlThe symmetric fourth order tensor, mathsfI^textsym, is the resulting tensor from taking the derivative of a symmetric second order tensor mathbfA^textsym with itself:mathsfI^textsym = fracpartial mathbfA^textsympartial mathbfA^textsym Leftrightarrow I^textsym_ijkl = fracpartial A^textsym_ijpartial A^textsym_kl = frac12 (delta_ik delta_jl + delta_il delta_jk)Identity tensors are created using the function one, applied to the type of tensor that should be created:julia> one(SymmetricTensor{2, 2})\n2×2 Tensors.SymmetricTensor{2,2,Float64,3}:\n 1.0  0.0\n 0.0  1.0A Julia Array with identity tensors can be created with ones, with the tensor type and dimensions of the array as arguments:julia> ones(Tensor{2,2}, 2, 2)\n2×2 Array{Tensors.Tensor{2,2,Float64,4},2}:\n [1.0 0.0; 0.0 1.0]  [1.0 0.0; 0.0 1.0]\n [1.0 0.0; 0.0 1.0]  [1.0 0.0; 0.0 1.0]"
},

{
    "location": "man/constructing_tensors.html#Tensors.basevec",
    "page": "Constructing tensors",
    "title": "Tensors.basevec",
    "category": "Function",
    "text": "basevec(::Type{Vec{dim, T}})\nbasevec(::Type{Vec{dim, T}}, i)\nbasevec(::Vec{dim, T})\nbasevec(::Vec{dim, T}, i)\n\nReturn a tuple with the base vectors corresponding to the dimension dim and type T. An optional integer i can be used to extract the i:th base vector. The alias eᵢ can also be used, written e\\_i<TAB>.\n\nExample:\n\njulia> eᵢ(Vec{2, Float64})\n([1.0, 0.0], [0.0, 1.0])\n\njulia> eᵢ(Vec{2, Float64}, 2)\n2-element Tensors.Tensor{1,2,Float64,2}:\n 0.0\n 1.0\n\n\n\n"
},

{
    "location": "man/constructing_tensors.html#Base-vectors-1",
    "page": "Constructing tensors",
    "title": "Base vectors",
    "category": "section",
    "text": "basevec"
},

{
    "location": "man/constructing_tensors.html#From-arrays-/-tuples-1",
    "page": "Constructing tensors",
    "title": "From arrays / tuples",
    "category": "section",
    "text": "Tensors can also be created from a tuple or an array with the same number of elements as the number of independent indices in the tensor. For example, a first order tensor (vector) in two dimensions is here created from a vector of length two:julia> Tensor{1,2}([1.0,2.0])\n2-element Tensors.Tensor{1,2,Float64,2}:\n 1.0\n 2.0Below, a second order symmetric tensor in two dimensions is created from a tuple. Since the number of independent indices in this tensor is three, the length of the tuple is also three. For symmetric tensors, the order of the numbers in the input tuple is column by column, starting at the diagonal.julia> SymmetricTensor{2,2}((1.0,2.0,3.0))\n2×2 Tensors.SymmetricTensor{2,2,Float64,3}:\n 1.0  2.0\n 2.0  3.0"
},

{
    "location": "man/constructing_tensors.html#function_index-1",
    "page": "Constructing tensors",
    "title": "From a function",
    "category": "section",
    "text": "A tensor can be created from a function f(indices...) -> v which maps a set of indices to a value. The number of arguments of the function should be equal to the order of the tensor.julia> SymmetricTensor{2,2,Float64}((i,j) -> i + j)\n2×2 Tensors.SymmetricTensor{2,2,Float64,3}:\n 2.0  3.0\n 3.0  4.0For symmetric tensors, the function is only called for the lower triangular part."
},

{
    "location": "man/constructing_tensors.html#Diagonal-tensors-1",
    "page": "Constructing tensors",
    "title": "Diagonal tensors",
    "category": "section",
    "text": "A diagonal second order tensor can be created by either giving a number or a vector that should appear on the diagonal:julia> diagm(Tensor{2,2}, 2.0)\n2×2 Tensors.Tensor{2,2,Float64,4}:\n 2.0  0.0\n 0.0  2.0\n\njulia> diagm(SymmetricTensor{2,3}, [1.0, 2.0, 3.0])\n3×3 Tensors.SymmetricTensor{2,3,Float64,6}:\n 1.0  0.0  0.0\n 0.0  2.0  0.0\n 0.0  0.0  3.0"
},

{
    "location": "man/constructing_tensors.html#Converting-to-tensors-1",
    "page": "Constructing tensors",
    "title": "Converting to tensors",
    "category": "section",
    "text": "Sometimes it is necessary to convert between standard Julia Array's and Tensor's. When the number type is a bits type (like for floats or integers) this is conveniently done by the reinterpret function. For example, a 2×5 Julia Array can be translated to a vector of Vec{2} with the following codejulia> data = rand(2, 5)\n2×5 Array{Float64,2}:\n 0.590845  0.566237  0.794026  0.200586  0.246837\n 0.766797  0.460085  0.854147  0.298614  0.579672\n\njulia> tensor_data = reinterpret(Vec{2, Float64}, data, (5,))\n5-element Array{Tensors.Tensor{1,2,Float64,2},1}:\n [0.590845, 0.766797]\n [0.566237, 0.460085]\n [0.794026, 0.854147]\n [0.200586, 0.298614]\n [0.246837, 0.579672]The data can also be reinterpreted back to a Julia Arrayjulia> data = reinterpret(Float64, tensor_data, (2,5))\n2×5 Array{Float64,2}:\n 0.590845  0.566237  0.794026  0.200586  0.246837\n 0.766797  0.460085  0.854147  0.298614  0.579672"
},

{
    "location": "man/indexing.html#",
    "page": "Indexing",
    "title": "Indexing",
    "category": "page",
    "text": "DocTestSetup = quote\n    srand(1234)\n    using Tensors\nend"
},

{
    "location": "man/indexing.html#Indexing-1",
    "page": "Indexing",
    "title": "Indexing",
    "category": "section",
    "text": "Indexing into a (Symmetric)Tensor{dim, order} is performed like for an Array of dimension order.julia> A = rand(Tensor{2, 2});\n\njulia> A[1, 2]\n0.5662374165061859\n\njulia> B = rand(SymmetricTensor{4, 2});\n\njulia> B[1, 2, 1, 2]\n0.24683718661000897Slicing will produce a Tensor of lower order.julia> A = rand(Tensor{2, 2});\n\njulia> A[:, 1]\n2-element Tensors.Tensor{1,2,Float64,2}:\n 0.590845\n 0.766797Since Tensors are immutable there is no setindex! function defined on them. Instead, use the functionality to create tensors from functions as described here. As an example, this sets the [1,2] index on a tensor to one and the rest to zero:julia> Tensor{2, 2}((i,j) -> i == 1 && j == 2 ? 1.0 : 0.0)\n2×2 Tensors.Tensor{2,2,Float64,4}:\n 0.0  1.0\n 0.0  0.0For symmetric tensors, note that you should only set the lower triangular part of the tensor:julia> SymmetricTensor{2, 2}((i,j) -> i == 1 && j == 2 ? 1.0 : 0.0)\n2×2 Tensors.SymmetricTensor{2,2,Float64,3}:\n 0.0  0.0\n 0.0  0.0\n\njulia> SymmetricTensor{2, 2}((i,j) -> i == 2 && j == 1 ? 1.0 : 0.0)\n2×2 Tensors.SymmetricTensor{2,2,Float64,3}:\n 0.0  1.0\n 1.0  0.0"
},

{
    "location": "man/binary_operators.html#",
    "page": "Binary Operations",
    "title": "Binary Operations",
    "category": "page",
    "text": "DocTestSetup = quote\n    srand(1234)\n    using Tensors\nend"
},

{
    "location": "man/binary_operators.html#Binary-Operations-1",
    "page": "Binary Operations",
    "title": "Binary Operations",
    "category": "section",
    "text": "Pages = [\"binary_operators.md\"]"
},

{
    "location": "man/binary_operators.html#Base.LinAlg.dot",
    "page": "Binary Operations",
    "title": "Base.LinAlg.dot",
    "category": "Function",
    "text": "dot(::Vec, ::Vec)\ndot(::Vec, ::SecondOrderTensor)\ndot(::SecondOrderTensor, ::Vec)\ndot(::SecondOrderTensor, ::SecondOrderTensor)\n\nComputes the dot product (single contraction) between two tensors. The symbol ⋅, written \\cdot, is overloaded for single contraction.\n\nExample:\n\njulia> A = rand(Tensor{2, 2})\n2×2 Tensors.Tensor{2,2,Float64,4}:\n 0.590845  0.566237\n 0.766797  0.460085\n\njulia> B = rand(Tensor{1, 2})\n2-element Tensors.Tensor{1,2,Float64,2}:\n 0.794026\n 0.854147\n\njulia> dot(A, B)\n2-element Tensors.Tensor{1,2,Float64,2}:\n 0.952796\n 1.00184\n\njulia> A ⋅ B\n2-element Tensors.Tensor{1,2,Float64,2}:\n 0.952796\n 1.00184\n\n\n\n"
},

{
    "location": "man/binary_operators.html#Dot-product-(single-contraction)-1",
    "page": "Binary Operations",
    "title": "Dot product (single contraction)",
    "category": "section",
    "text": "The dot product (or single contraction) between a tensor of order n and a tensor of order m is a tensor of order m + n - 2. For example, single contraction between two vectors mathbfb and mathbfc can be written as:a = mathbfb cdot mathbfc Leftrightarrow a = b_i c_iand single contraction between a second order tensor mathbfB and a vector mathbfc:mathbfa = mathbfB cdot mathbfc Leftrightarrow a_i = B_ij c_jdot"
},

{
    "location": "man/binary_operators.html#Tensors.dcontract",
    "page": "Binary Operations",
    "title": "Tensors.dcontract",
    "category": "Function",
    "text": "dcontract(::SecondOrderTensor, ::SecondOrderTensor)\ndcontract(::SecondOrderTensor, ::FourthOrderTensor)\ndcontract(::FourthOrderTensor, ::SecondOrderTensor)\ndcontract(::FourthOrderTensor, ::FourthOrderTensor)\n\nComputes the double contraction between two tensors. The symbol ⊡, written \\boxdot, is overloaded for double contraction. The reason : is not used is because it does not have the same precedence as multiplication.\n\nExample:\n\njulia> A = rand(SymmetricTensor{2, 2});\n\njulia> B = rand(SymmetricTensor{2, 2});\n\njulia> dcontract(A,B)\n1.9732018397544984\n\njulia> A ⊡ B\n1.9732018397544984\n\n\n\n"
},

{
    "location": "man/binary_operators.html#Double-contraction-1",
    "page": "Binary Operations",
    "title": "Double contraction",
    "category": "section",
    "text": "A double contraction between two tensors contracts the two most inner indices. The result of a double contraction between a tensor of order n and a tensor of order m is a tensor of order m + n - 4. For example, double contraction between two second order tensors mathbfB and mathbfC can be written as:a = mathbfB  mathbfC Leftrightarrow a = B_ij C_ijand double contraction between a fourth order tensor mathsfB and a second order tensor mathbfC:mathbfA = mathsfB  mathbfC Leftrightarrow A_ij = B_ijkl C_kldcontract"
},

{
    "location": "man/binary_operators.html#Tensors.otimes",
    "page": "Binary Operations",
    "title": "Tensors.otimes",
    "category": "Function",
    "text": "otimes(::Vec, ::Vec)\notimes(::SecondOrderTensor, ::SecondOrderTensor)\n\nComputes the open product between two tensors. The symbol ⊗, written \\otimes, is overloaded for tensor products.\n\nExample:\n\njulia> A = rand(SymmetricTensor{2, 2});\n\njulia> B = rand(SymmetricTensor{2, 2});\n\njulia> A ⊗ B\n2×2×2×2 Tensors.SymmetricTensor{4,2,Float64,9}:\n[:, :, 1, 1] =\n 0.271839  0.352792\n 0.352792  0.260518\n\n[:, :, 2, 1] =\n 0.469146  0.608857\n 0.608857  0.449607\n\n[:, :, 1, 2] =\n 0.469146  0.608857\n 0.608857  0.449607\n\n[:, :, 2, 2] =\n 0.504668  0.654957\n 0.654957  0.48365\n\n\n\n"
},

{
    "location": "man/binary_operators.html#Tensor-product-(open-product)-1",
    "page": "Binary Operations",
    "title": "Tensor product (open product)",
    "category": "section",
    "text": "The tensor product (or open product) between a tensor of order n and a tensor of order m is a tensor of order m + n.  For example, open product between two vectors mathbfb and mathbfc can be written as:mathbfA = mathbfb otimes mathbfc Leftrightarrow A_ij = b_i c_jand open product between two second order tensors mathbfB and mathbfC:mathsfA = mathbfB otimes mathbfC Leftrightarrow A_ijkl = B_ij C_klotimes"
},

{
    "location": "man/other_operators.html#",
    "page": "Other operators",
    "title": "Other operators",
    "category": "page",
    "text": "DocTestSetup = quote\n    srand(1234)\n    using Tensors\nend"
},

{
    "location": "man/other_operators.html#Other-operators-1",
    "page": "Other operators",
    "title": "Other operators",
    "category": "section",
    "text": "Pages = [\"other_operators.md\"]"
},

{
    "location": "man/other_operators.html#Tensors.tdot",
    "page": "Other operators",
    "title": "Tensors.tdot",
    "category": "Function",
    "text": "tdot(::SecondOrderTensor)\n\nComputes the transpose-dot of a second order tensor with itself. Returns a SymmetricTensor.\n\nExample:\n\njulia> A = rand(Tensor{2,3})\n3×3 Tensors.Tensor{2,3,Float64,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> tdot(A)\n3×3 Tensors.SymmetricTensor{2,3,Float64,6}:\n 1.2577   1.36435   0.48726\n 1.36435  1.57172   0.540229\n 0.48726  0.540229  0.190334\n\n\n\n"
},

{
    "location": "man/other_operators.html#Tensors.dott",
    "page": "Other operators",
    "title": "Tensors.dott",
    "category": "Function",
    "text": "dott(::SecondOrderTensor)\n\nComputes the dot-transpose of a second order tensor with itself. Returns a SymmetricTensor.\n\nExample:\n\njulia> A = rand(Tensor{2,3})\n3×3 Tensors.Tensor{2,3,Float64,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> dott(A)\n3×3 Tensors.SymmetricTensor{2,3,Float64,6}:\n 0.601011  0.878275  0.777051\n 0.878275  1.30763   1.18611\n 0.777051  1.18611   1.11112\n\n\n\n"
},

{
    "location": "man/other_operators.html#Transpose-dot-1",
    "page": "Other operators",
    "title": "Transpose-dot",
    "category": "section",
    "text": "The dot product between the transpose of a tensor with itself. Results in a symmetric tensor.mathbfA = mathbfB^textT cdot mathbfB Leftrightarrow A_ij = B_ki^textT B_kj = B_ik B_kjmathbfA = mathbfB cdot mathbfB^textT Leftrightarrow A_ij = B_ik B_jk^textT = B_ik B_kjtdot\ndott"
},

{
    "location": "man/other_operators.html#Base.LinAlg.norm",
    "page": "Other operators",
    "title": "Base.LinAlg.norm",
    "category": "Function",
    "text": "norm(::Vec)\nnorm(::SecondOrderTensor)\nnorm(::FourthOrderTensor)\n\nComputes the norm of a tensor.\n\nExample:\n\njulia> A = rand(Tensor{2,3})\n3×3 Tensors.Tensor{2,3,Float64,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> norm(A)\n1.7377443667834924\n\n\n\n"
},

{
    "location": "man/other_operators.html#Norm-1",
    "page": "Other operators",
    "title": "Norm",
    "category": "section",
    "text": "The (2)-norm of a tensor is defined for a vector, second order tensor and fourth order tensor asmathbfa = sqrtmathbfa cdot mathbfa Leftrightarrow a_i = sqrta_i a_imathbfA = sqrtmathbfA  mathbfA Leftrightarrow A_ij = sqrtA_ij A_ijmathsfA = sqrtmathsfA  mathsfA Leftrightarrow A_ijkl = sqrtA_ijkl A_ijklnorm"
},

{
    "location": "man/other_operators.html#Base.LinAlg.trace",
    "page": "Other operators",
    "title": "Base.LinAlg.trace",
    "category": "Function",
    "text": "trace(::SecondOrderTensor)\n\nComputes the trace of a second order tensor. The synonym vol can also be used.\n\nExample:\n\njulia> A = rand(SymmetricTensor{2,3})\n3×3 Tensors.SymmetricTensor{2,3,Float64,6}:\n 0.590845  0.766797  0.566237\n 0.766797  0.460085  0.794026\n 0.566237  0.794026  0.854147\n\njulia> trace(A)\n1.9050765715072775\n\n\n\n"
},

{
    "location": "man/other_operators.html#Trace-1",
    "page": "Other operators",
    "title": "Trace",
    "category": "section",
    "text": "The trace for a second order tensor is defined as the sum of the diagonal elements. This can be written astexttr(mathbfA) = mathbfI  mathbfA Leftrightarrow texttr(A_ij) = A_iitrace"
},

{
    "location": "man/other_operators.html#Base.LinAlg.det",
    "page": "Other operators",
    "title": "Base.LinAlg.det",
    "category": "Function",
    "text": "det(::SecondOrderTensor)\n\nComputes the determinant of a second order tensor.\n\nExample:\n\njulia> A = rand(SymmetricTensor{2,3})\n3×3 Tensors.SymmetricTensor{2,3,Float64,6}:\n 0.590845  0.766797  0.566237\n 0.766797  0.460085  0.794026\n 0.566237  0.794026  0.854147\n\njulia> det(A)\n-0.1005427219925894\n\n\n\n"
},

{
    "location": "man/other_operators.html#Determinant-1",
    "page": "Other operators",
    "title": "Determinant",
    "category": "section",
    "text": "Determinant for a second order tensor.det"
},

{
    "location": "man/other_operators.html#Base.inv",
    "page": "Other operators",
    "title": "Base.inv",
    "category": "Function",
    "text": "inv(::SecondOrderTensor)\n\nComputes the inverse of a second order tensor.\n\nExample:\n\njulia> A = rand(Tensor{2,3})\n3×3 Tensors.Tensor{2,3,Float64,9}:\n 0.590845  0.460085  0.200586\n 0.766797  0.794026  0.298614\n 0.566237  0.854147  0.246837\n\njulia> inv(A)\n3×3 Tensors.Tensor{2,3,Float64,9}:\n  19.7146   -19.2802    7.30384\n   6.73809  -10.7687    7.55198\n -68.541     81.4917  -38.8361\n\n\n\n"
},

{
    "location": "man/other_operators.html#Inverse-1",
    "page": "Other operators",
    "title": "Inverse",
    "category": "section",
    "text": "Inverse of a second order tensor such thatmathbfA^-1 cdot mathbfA = mathbfIwhere mathbfI is the second order identitiy tensor.inv"
},

{
    "location": "man/other_operators.html#Base.transpose",
    "page": "Other operators",
    "title": "Base.transpose",
    "category": "Function",
    "text": "transpose(::Vec)\ntranspose(::SecondOrderTensor)\ntranspose(::FourthOrderTensor)\n\nComputes the transpose of a tensor. For a fourth order tensor, the transpose is the minor transpose.\n\nExample:\n\njulia> A = rand(Tensor{2,2})\n2×2 Tensors.Tensor{2,2,Float64,4}:\n 0.590845  0.566237\n 0.766797  0.460085\n\njulia> A'\n2×2 Tensors.Tensor{2,2,Float64,4}:\n 0.590845  0.766797\n 0.566237  0.460085\n\n\n\n"
},

{
    "location": "man/other_operators.html#Tensors.minortranspose",
    "page": "Other operators",
    "title": "Tensors.minortranspose",
    "category": "Function",
    "text": "minortranspose(::FourthOrderTensor)\n\nComputes the minor transpose of a fourth order tensor.\n\n\n\n"
},

{
    "location": "man/other_operators.html#Tensors.majortranspose",
    "page": "Other operators",
    "title": "Tensors.majortranspose",
    "category": "Function",
    "text": "majortranspose(::FourthOrderTensor)\n\nComputes the major transpose of a fourth order tensor.\n\n\n\n"
},

{
    "location": "man/other_operators.html#Transpose-1",
    "page": "Other operators",
    "title": "Transpose",
    "category": "section",
    "text": "Transpose of tensors is defined by changing the order of the tensor's \"legs\". The transpose of a vector/symmetric tensor is the vector/tensor itself. The transpose of a second order tensor can be written as:A_ij^textT = A_jiand for a fourth order tensor the minor transpose can be written asA_ijkl^textt = A_jilkand the major transpose asA_ijkl^textT = A_klijtranspose\nminortranspose\nmajortranspose"
},

{
    "location": "man/other_operators.html#Tensors.symmetric",
    "page": "Other operators",
    "title": "Tensors.symmetric",
    "category": "Function",
    "text": "symmetric(::SecondOrderTensor)\nsymmetric(::FourthOrderTensor)\n\nComputes the symmetric part of a second or fourth order tensor. For a fourth order tensor, the symmetric part is the same as the minor symmetric part. Returns a SymmetricTensor.\n\nExample:\n\njulia> A = rand(Tensor{2,2})\n2×2 Tensors.Tensor{2,2,Float64,4}:\n 0.590845  0.566237\n 0.766797  0.460085\n\njulia> symmetric(A)\n2×2 Tensors.SymmetricTensor{2,2,Float64,3}:\n 0.590845  0.666517\n 0.666517  0.460085\n\n\n\n"
},

{
    "location": "man/other_operators.html#Tensors.minorsymmetric",
    "page": "Other operators",
    "title": "Tensors.minorsymmetric",
    "category": "Function",
    "text": "minorsymmetric(::FourthOrderTensor)\n\nComputes the minor symmetric part of a fourth order tensor, returns a SymmetricTensor{4}.\n\n\n\n"
},

{
    "location": "man/other_operators.html#Tensors.majorsymmetric",
    "page": "Other operators",
    "title": "Tensors.majorsymmetric",
    "category": "Function",
    "text": "majorsymmetric(::FourthOrderTensor)\n\nComputes the major symmetric part of a fourth order tensor, returns a Tensor{4}.\n\n\n\n"
},

{
    "location": "man/other_operators.html#Symmetric-1",
    "page": "Other operators",
    "title": "Symmetric",
    "category": "section",
    "text": "The symmetric part of a second and fourth order tensor is defined by:mathbfA^textsym = frac12(mathbfA + mathbfA^textT) Leftrightarrow A_ij^textsym = frac12(A_ij + A_ji)mathsfA^textsym = frac12(mathsfA + mathsfA^textt) Leftrightarrow A_ijkl^textsym = frac12(A_ijkl + A_jilk)symmetric\nminorsymmetric\nmajorsymmetric"
},

{
    "location": "man/other_operators.html#Tensors.skew",
    "page": "Other operators",
    "title": "Tensors.skew",
    "category": "Function",
    "text": "skew(::SecondOrderTensor)\n\nComputes the skew-symmetric (anti-symmetric) part of a second order tensor, returns a Tensor{2}.\n\n\n\n"
},

{
    "location": "man/other_operators.html#Skew-symmetric-1",
    "page": "Other operators",
    "title": "Skew symmetric",
    "category": "section",
    "text": "The skew symmetric part of a second order tensor is defined bymathbfA^textskw = frac12(mathbfA - mathbfA^textT) Leftrightarrow A^textskw_ij = frac12(A_ij - A_ji)The skew symmetric part of a symmetric tensor is zero.skew"
},

{
    "location": "man/other_operators.html#Tensors.dev",
    "page": "Other operators",
    "title": "Tensors.dev",
    "category": "Function",
    "text": "dev(::SecondOrderTensor)\n\nComputes the deviatoric part of a second order tensor.\n\nExample:\n\njulia> A = rand(Tensor{2,3});\n\njulia> dev(A)\n3×3 Tensors.Tensor{2,3,Float64,9}:\n 0.0469421  0.460085   0.200586\n 0.766797   0.250123   0.298614\n 0.566237   0.854147  -0.297065\n\njulia> trace(dev(A))\n0.0\n\n\n\n"
},

{
    "location": "man/other_operators.html#Deviator-1",
    "page": "Other operators",
    "title": "Deviator",
    "category": "section",
    "text": "The deviatoric part of a second order tensor is defined bymathbfA^textdev = mathbfA - frac13 mathrmtracemathbfA mathbfI Leftrightarrow A_ij^textdev = A_ij - frac13A_kkdelta_ijdev"
},

{
    "location": "man/other_operators.html#Base.LinAlg.cross",
    "page": "Other operators",
    "title": "Base.LinAlg.cross",
    "category": "Function",
    "text": "cross(::Vec, ::Vec)\n\nComputes the cross product between two Vec vectors, returns a Vec{3}. For dimensions 1 and 2 the Vec's are expanded to 3D first. The infix operator × (written \\times) can also be used.\n\nExample:\n\njulia> a = rand(Vec{3})\n3-element Tensors.Tensor{1,3,Float64,3}:\n 0.590845\n 0.766797\n 0.566237\n\njulia> b = rand(Vec{3})\n3-element Tensors.Tensor{1,3,Float64,3}:\n 0.460085\n 0.794026\n 0.854147\n\njulia> a × b\n3-element Tensors.Tensor{1,3,Float64,3}:\n  0.20535\n -0.24415\n  0.116354\n\n\n\n"
},

{
    "location": "man/other_operators.html#Cross-product-1",
    "page": "Other operators",
    "title": "Cross product",
    "category": "section",
    "text": "The cross product between two vectors is defined asmathbfa = mathbfb times mathbfc Leftrightarrow a_i = epsilon_ijk b_j c_kcross"
},

{
    "location": "man/other_operators.html#Base.LinAlg.eigfact",
    "page": "Other operators",
    "title": "Base.LinAlg.eigfact",
    "category": "Function",
    "text": "eigfact(::SymmetricTensor{2})\n\nCompute the eigenvalues and eigenvectors of a symmetric second order tensor and return an Eigen object. The eigenvalues are stored in a Vec, sorted in ascending order. The corresponding eigenvectors are stored as the columns of a Tensor.\n\nSee eigvals and eigvecs.\n\nExample:\n\njulia> A = rand(SymmetricTensor{2, 2})\n2×2 Tensors.SymmetricTensor{2,2,Float64,3}:\n 0.590845  0.766797\n 0.766797  0.566237\n\njulia> E = eigfact(A)\nTensors.Eigen{Float64,2,4}([-0.188355, 1.34544], [-0.701412 0.712756; 0.712756 0.701412])\n\njulia> eigvals(E)\n2-element Tensors.Tensor{1,2,Float64,2}:\n -0.188355\n  1.34544\n\njulia> eigvecs(E)\n2×2 Tensors.Tensor{2,2,Float64,4}:\n -0.701412  0.712756\n  0.712756  0.701412\n\n\n\n"
},

{
    "location": "man/other_operators.html#Base.LinAlg.eigvals",
    "page": "Other operators",
    "title": "Base.LinAlg.eigvals",
    "category": "Function",
    "text": "eigvals(::SymmetricTensor{2})\n\nCompute the eigenvalues of a symmetric second order tensor.\n\n\n\neigvals(::Eigen)\n\nExtract eigenvalues from an Eigen object, returned by eigfact.\n\n\n\n"
},

{
    "location": "man/other_operators.html#Base.LinAlg.eigvecs",
    "page": "Other operators",
    "title": "Base.LinAlg.eigvecs",
    "category": "Function",
    "text": "eigvecs(::SymmetricTensor{2})\n\nCompute the eigenvectors of a symmetric second order tensor.\n\n\n\neigvecs(::Eigen)\n\nExtract eigenvectors from an Eigen object, returned by eigfact.\n\n\n\n"
},

{
    "location": "man/other_operators.html#Base.LinAlg.eig",
    "page": "Other operators",
    "title": "Base.LinAlg.eig",
    "category": "Function",
    "text": "eig(::SymmetricTensor{2})\n\nCompute the eigenvalues and eigenvectors of a symmetric second order tensor. eig is a wrapper around eigfact which extracts eigenvalues and eigenvectors to a tuple.\n\nExample:\n\njulia> A = rand(SymmetricTensor{2, 2})\n2×2 Tensors.SymmetricTensor{2,2,Float64,3}:\n 0.590845  0.766797\n 0.766797  0.566237\n\njulia> Λ, Φ = eig(A);\n\njulia> Λ\n2-element Tensors.Tensor{1,2,Float64,2}:\n -0.188355\n  1.34544\n\njulia> Φ\n2×2 Tensors.Tensor{2,2,Float64,4}:\n -0.701412  0.712756\n  0.712756  0.701412\n\njulia> Φ ⋅ diagm(Tensor{2, 2}, Λ) ⋅ inv(Φ) # Same as A\n2×2 Tensors.Tensor{2,2,Float64,4}:\n 0.590845  0.766797\n 0.766797  0.566237\n\n\n\n"
},

{
    "location": "man/other_operators.html#Eigenvalues-and-eigenvectors-1",
    "page": "Other operators",
    "title": "Eigenvalues and eigenvectors",
    "category": "section",
    "text": "The eigenvalues and eigenvectors of a (symmetric) second order tensor, mathbfA can be solved from the eigenvalue problemmathbfA cdot mathbfv_i = lambda_i mathbfv_i qquad i = 1 dots textdimwhere lambda_i are the eigenvalues and mathbfv_i are the corresponding eigenvectors.eigfact\neigvals\neigvecs\neig"
},

{
    "location": "man/other_operators.html#Tensors.rotate",
    "page": "Other operators",
    "title": "Tensors.rotate",
    "category": "Function",
    "text": "Rotate a three dimensional vector x around another vector u a total of θ radians.\n\nrotate(x::Vec{3}, u::Vec{3}, θ::Number)\n\nExample:\n\njulia> x = Vec{3}((0.0, 0.0, 1.0))\n3-element Tensors.Tensor{1,3,Float64,3}:\n 0.0\n 0.0\n 1.0\n\njulia> u = Vec{3}((0.0, 1.0, 0.0))\n3-element Tensors.Tensor{1,3,Float64,3}:\n 0.0\n 1.0\n 0.0\n\njulia> rotate(x, u, π/2)\n3-element Tensors.Tensor{1,3,Float64,3}:\n 1.0\n 0.0\n 6.12323e-17\n\n\n\n"
},

{
    "location": "man/other_operators.html#Rotations-1",
    "page": "Other operators",
    "title": "Rotations",
    "category": "section",
    "text": "rotate"
},

{
    "location": "man/other_operators.html#Tensors.dotdot",
    "page": "Other operators",
    "title": "Tensors.dotdot",
    "category": "Function",
    "text": "dotdot(::Vec, ::SymmetricFourthOrderTensor, ::Vec)\n\nComputes a special dot product between two vectors and a symmetric fourth order tensor such that a_k C_ikjl b_l.\n\n\n\n"
},

{
    "location": "man/other_operators.html#Special-operations-1",
    "page": "Other operators",
    "title": "Special operations",
    "category": "section",
    "text": "For computing a special dot product between two vectors mathbfa and mathbfb with a fourth order symmetric tensor mathbfC such that a_k C_ikjl b_l there is dotdot(a, C, b). This function is useful because it is the expression for the tangent matrix in continuum mechanics when the displacements are approximated by scalar shape functions.dotdot"
},

{
    "location": "man/other_operators.html#Tensors.tovoigt",
    "page": "Other operators",
    "title": "Tensors.tovoigt",
    "category": "Function",
    "text": "tovoigt(A::Union{SecondOrderTensor, FourthOrderTensor}; offdiagscale)\n\nConverts a tensor to \"Voigt\"-format using the following index order: [11, 22, 33, 23, 13, 12, 32, 31, 21]. For SymmetricTensors, the keyword argument offdiagscale sets a scaling factor on the offdiagonal elements\n\nSee also fromvoigt.\n\njulia> tovoigt(Tensor{2,3}(1:9))\n9-element Array{Int64,1}:\n 1\n 5\n 9\n 8\n 7\n 4\n 6\n 3\n 2\n\njulia> tovoigt(SymmetricTensor{2,3}(1.0:1.0:6.0); offdiagscale = 2.0)\n6-element Array{Float64,1}:\n  1.0\n  4.0\n  6.0\n 10.0\n  6.0\n  4.0\n\njulia> tovoigt(Tensor{4,2}(1:16))\n4×4 Array{Int64,2}:\n 1  13   9  5\n 4  16  12  8\n 3  15  11  7\n 2  14  10  6\n\n\n\n"
},

{
    "location": "man/other_operators.html#Tensors.fromvoigt",
    "page": "Other operators",
    "title": "Tensors.fromvoigt",
    "category": "Function",
    "text": "fromvoigt(::T, A::Array)\n\nConverts an array A stored in Voigt format to a Tensor of type T. For SymmetricTensors, the keyword argument offdiagscale sets an inverse scaling factor on the offdiagonal elements.\n\nSee also tovoigt.\n\njulia> fromvoigt(Tensor{2,3}, 1.0:1.0:9.0)\n3×3 Tensors.Tensor{2,3,Float64,9}:\n 1.0  6.0  5.0\n 9.0  2.0  4.0\n 8.0  7.0  3.0\n\n\n\n"
},

{
    "location": "man/other_operators.html#Voigt-1",
    "page": "Other operators",
    "title": "Voigt",
    "category": "section",
    "text": "tovoigt\nfromvoigt"
},

{
    "location": "man/storing_tensors.html#",
    "page": "Storing tensors",
    "title": "Storing tensors",
    "category": "page",
    "text": ""
},

{
    "location": "man/storing_tensors.html#Storing-tensors-1",
    "page": "Storing tensors",
    "title": "Storing tensors",
    "category": "section",
    "text": "Even though a user mostly deals with the Tensor{order, dim, T} parameters, the full parameter list for a tensor is actually Tensor{order, dim, T, N} where N is the number of independent elements in the tensor. The reason for this is that the internal storage for tensors is a NTuple{N, T}. In order to get good performance when storing tensors in other types it is important that the container type is also parametrized on N. For example, when storing one symmetric second order tensor and one unsymmetric tensor, this is the preferred way:immutable Container{dim, T, N, M}\n    sym_tens::SymmetricTensor{2, dim, T, N}\n    tens::Tensor{2, dim, T, M}\nendLeaving out the M and N would lead to bad performance.tip: Tip\nThe number of independent elements N are already included in the typealias Vec so they can be stored with e.g.immutable VecContainer{dim, T}\n    vec::Vec{dim, T}\nendwithout causing bad performance."
},

{
    "location": "man/automatic_differentiation.html#",
    "page": "Automatic Differentiation",
    "title": "Automatic Differentiation",
    "category": "page",
    "text": "DocTestSetup = quote\n    srand(1234)\n    using Tensors\nend"
},

{
    "location": "man/automatic_differentiation.html#Base.LinAlg.gradient",
    "page": "Automatic Differentiation",
    "title": "Base.LinAlg.gradient",
    "category": "Function",
    "text": "gradient(f::Function, v::Union{SecondOrderTensor, Vec})\ngradient(f::Function, v::Union{SecondOrderTensor, Vec}, :all)\n\nComputes the gradient of the input function. If the (pseudo)-keyword all is given, the value of the function is also returned as a second output argument.\n\nExample:\n\njulia> A = rand(SymmetricTensor{2, 2});\n\njulia> ∇f = gradient(norm, A)\n2×2 Tensors.SymmetricTensor{2,2,Float64,3}:\n 0.434906  0.56442\n 0.56442   0.416793\n\njulia> ∇f, f = gradient(norm, A, :all);\n\n\n\n"
},

{
    "location": "man/automatic_differentiation.html#Tensors.hessian",
    "page": "Automatic Differentiation",
    "title": "Tensors.hessian",
    "category": "Function",
    "text": "hessian(f::Function, v::Union{SecondOrderTensor, Vec})\nhessian(f::Function, v::Union{SecondOrderTensor, Vec}, :all)\n\nComputes the hessian of the input function. If the (pseudo)-keyword all is given, the lower order results (gradient and value) of the function is also returned as a second and third output argument.\n\nExample:\n\njulia> A = rand(SymmetricTensor{2, 2});\n\njulia> ∇∇f = hessian(norm, A)\n2×2×2×2 Tensors.SymmetricTensor{4,2,Float64,9}:\n[:, :, 1, 1] =\n  0.596851  -0.180684\n -0.180684  -0.133425\n\n[:, :, 2, 1] =\n -0.180684   0.133546\n  0.133546  -0.173159\n\n[:, :, 1, 2] =\n -0.180684   0.133546\n  0.133546  -0.173159\n\n[:, :, 2, 2] =\n -0.133425  -0.173159\n -0.173159   0.608207\n\njulia> ∇∇f, ∇f, f = hessian(norm, A, :all);\n\n\n\n"
},

{
    "location": "man/automatic_differentiation.html#Automatic-Differentiation-1",
    "page": "Automatic Differentiation",
    "title": "Automatic Differentiation",
    "category": "section",
    "text": "Pages = [\"automatic_differentiation.md\"]Tensors supports forward mode automatic differentiation (AD) of tensorial functions to compute first order derivatives (gradients) and second order derivatives (Hessians). It does this by exploiting the Dual number defined in ForwardDiff.jl. While ForwardDiff.jl can itself be used to differentiate tensor functions it is a bit awkward because ForwardDiff.jl is written to work with standard Julia Arrays. One therefore has to send the input argument as an Array to ForwardDiff.jl, convert it to a Tensor and then convert the output Array to a Tensor again. This can also be inefficient since these Arrays are allocated on the heap so one needs to preallocate which can be annoying.Instead, it is simpler to use Tensors own AD API to do the differentiation. This does not require any conversions and everything will be stack allocated so there is no need to preallocate.The API for AD in Tensors is gradient(f, A) and hessian(f, A) where f is a function and A is a first or second order tensor. For gradient the function can return a scalar, vector (in case the input is a vector) or a second order tensor. For hessian the function should return a scalar.When evaluating the function with dual numbers, the value (value and gradient in the case of hessian) is obtained automatically, along with the gradient. To obtain the lower order results gradient and hessian accepts a third arguement, a Symbol. Note that the symbol is only used to dispatch to the correct function, and thus it can be any symbol. In the examples the symbol :all is used to obtain all the lower order derivatives and values.gradient\nhessian"
},

{
    "location": "man/automatic_differentiation.html#Examples-1",
    "page": "Automatic Differentiation",
    "title": "Examples",
    "category": "section",
    "text": "We here give a few examples of differentiating various functions and compare with the analytical solution."
},

{
    "location": "man/automatic_differentiation.html#Norm-of-a-vector-1",
    "page": "Automatic Differentiation",
    "title": "Norm of a vector",
    "category": "section",
    "text": "f(mathbfx) = mathbfx quad Rightarrow quad partial f  partial mathbfx = mathbfx  mathbfxjulia> x = rand(Vec{2});\n\njulia> gradient(norm, x)\n2-element Tensors.Tensor{1,2,Float64,2}:\n 0.61036\n 0.792124\n\njulia> x / norm(x)\n2-element Tensors.Tensor{1,2,Float64,2}:\n 0.61036\n 0.792124"
},

{
    "location": "man/automatic_differentiation.html#Determinant-of-a-second-order-symmetric-tensor-1",
    "page": "Automatic Differentiation",
    "title": "Determinant of a second order symmetric tensor",
    "category": "section",
    "text": "f(mathbfA) = det mathbfA quad Rightarrow quad partial f  partial mathbfA = mathbfA^-T det mathbfAjulia> A = rand(SymmetricTensor{2,2});\n\njulia> gradient(det, A)\n2×2 Tensors.SymmetricTensor{2,2,Float64,3}:\n  0.566237  -0.766797\n -0.766797   0.590845\n\njulia> inv(A)' * det(A)\n2×2 Tensors.SymmetricTensor{2,2,Float64,3}:\n  0.566237  -0.766797\n -0.766797   0.590845"
},

{
    "location": "man/automatic_differentiation.html#Hessian-of-a-quadratic-potential-1",
    "page": "Automatic Differentiation",
    "title": "Hessian of a quadratic potential",
    "category": "section",
    "text": "psi(mathbfe) = 12 mathbfe  mathsfE  mathbfe quad Rightarrow quad partial psi  (partial mathbfe otimes partial mathbfe) = mathsfE^textsymwhere mathsfE^textsym is the major symmetric part of mathsfE.julia> E = rand(Tensor{4,2});\n\njulia> ψ(ϵ) = 1/2 * ϵ ⊡ E ⊡ ϵ;\n\njulia> E_sym = hessian(ψ, rand(Tensor{2,2}));\n\njulia> norm(majorsymmetric(E) - E_sym)\n0.0"
},

{
    "location": "demos.html#",
    "page": "Demos",
    "title": "Demos",
    "category": "page",
    "text": ""
},

{
    "location": "demos.html#Demos-1",
    "page": "Demos",
    "title": "Demos",
    "category": "section",
    "text": "This section contain a few demos of applying Tensors to continuum mechanics."
},

{
    "location": "demos.html#Creating-the-linear-elasticity-tensor-1",
    "page": "Demos",
    "title": "Creating the linear elasticity tensor",
    "category": "section",
    "text": "The linear elasticity tensor mathbfC can be defined from the Lamé parameters lambda and mu by the expression$ \\mathbf{C}_{ijkl} = \\lambda \\delta_{ij}\\delta_{kl} + \\mu(\\delta_{ij}\\delta_{jl} + \\delta_{il}\\delta_{jk}),$where delta_ij = 1 if i = j otherwise 0. It can also be computed in terms of the Young's modulus E and Poisson's ratio nu by the conversion formulas lambda = Enu  (1 + nu)(1 - 2nu) and mu = E  2(1 + nu)The code below creates the elasticity tensor for given parameters E and nu and dimension textttdim. Note the similarity between the mathematical formula and the code.using Tensors\nE = 200e9\nν = 0.3\ndim = 2\nλ = E*ν / ((1 + ν) * (1 - 2ν))\nμ = E / (2(1 + ν))\nδ(i,j) = i == j ? 1.0 : 0.0\nf = (i,j,k,l) -> λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))\n\nC = SymmetricTensor{4, dim}(f)"
},

{
    "location": "demos.html#Nonlinear-elasticity-material-1",
    "page": "Demos",
    "title": "Nonlinear elasticity material",
    "category": "section",
    "text": "For a deformation gradient mathbfF = mathbfI + nabla otimes mathbfu, where mathbfu is the deformation from the reference to the current configuration, the right Cauchy-Green deformation tensor is defined by mathbfC = mathbfF^T cdot mathbfF. The Second Piola Krichoff stress tensor mathbfS is derived from the Helmholtz free energy Psi by the relation mathbfS = 2 fracpartial Psipartial mathbfC.We can define potential energy of the material asPsi(mathbfC) = 12 mu (mathrmtr(hatmathbfC) - 3) + K_b(J-1)^2where hatmathbfC = mathrmdet(mathbfC)^-13 mathbfC and J = det(mathbfF) = sqrtdet(mathbfC) and the shear and bulk modulus are given by mu and K_b respectively.This free energy function can be implemented in Tensors as:function Ψ(C, μ, Kb)\n    detC = det(C)\n    J = sqrt(detC)\n    Ĉ = detC^(-1/3)*C\n    return 1/2*(μ * (trace(Ĉ)- 3) + Kb*(J-1)^2)\nendThe analytical expression for the Second Piola Kirchoff tensor is$ \\mathbf{S} = \\mu \\det(\\mathbf{C})^{-1/3}(\\mathbf{I} - 1/3 \\mathrm{tr}(\\mathbf{C})\\mathbf{C}^{-1}) + K_b(J-1)J\\mathbf{C}^{-1} $which can be implemented by the functionfunction S(C, μ, Kb)\n    I = one(C)\n    J = sqrt(det(C))\n    invC = inv(C)\n    return μ * det(C)^(-1/3)*(I - 1/3*trace(C)*invC) + Kb*(J-1)*J*invC\nend"
},

{
    "location": "demos.html#Automatic-differentiation-1",
    "page": "Demos",
    "title": "Automatic differentiation",
    "category": "section",
    "text": "For some material models it can be cumbersome to compute the analytical expression for the Second Piola Kirchoff tensor. We can then instead use Automatic Differentiation (AD). Below is an example which computes the Second Piola Kirchoff tensor using AD and compares it to the analytical answer.DocTestSetup = quote\n    srand(1234)\n    using Tensors\n    E = 200e9\n    ν = 0.3\n    dim = 2\n    λ = E*ν / ((1 + ν) * (1 - 2ν))\n    μ = E / (2(1 + ν))\n    δ(i,j) = i == j ? 1.0 : 0.0\n    f = (i,j,k,l) -> λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))\n\n    C = SymmetricTensor{4, dim}(f)\n\n    function Ψ(C, μ, Kb)\n        detC = det(C)\n        J = sqrt(detC)\n        Ĉ = detC^(-1/3)*C\n        return 1/2*(μ * (trace(Ĉ)- 3) + Kb*(J-1)^2)\n    end\n\n    function S(C, μ, Kb)\n        I = one(C)\n        J = sqrt(det(C))\n        invC = inv(C)\n        return μ * det(C)^(-1/3)*(I - 1/3*trace(C)*invC) + Kb*(J-1)*J*invC\n    end\nendjulia> μ = 1e10;\n\njulia> Kb = 1.66e11;\n\njulia> F = one(Tensor{2,3}) + rand(Tensor{2,3});\n\njulia> C = tdot(F);\n\njulia> S_AD = 2 * gradient(C -> Ψ(C, μ, Kb), C)\n3×3 Tensors.SymmetricTensor{2,3,Float64,6}:\n  4.30534e11  -2.30282e11  -8.52861e10\n -2.30282e11   4.38793e11  -2.64481e11\n -8.52861e10  -2.64481e11   7.85515e11\n\njulia> S(C, μ, Kb)\n3×3 Tensors.SymmetricTensor{2,3,Float64,6}:\n  4.30534e11  -2.30282e11  -8.52861e10\n -2.30282e11   4.38793e11  -2.64481e11\n -8.52861e10  -2.64481e11   7.85515e11"
},

]}
