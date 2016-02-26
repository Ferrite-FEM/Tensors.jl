function matmul!(C, A, B, BLKSIZE)
    fill!(C, 0.0)
    A_block = zeros(BLKSIZE, BLKSIZE)
    B_block = zeros(BLKSIZE, BLKSIZE)
    C_block = zeros(BLKSIZE, BLKSIZE)
    n = size(A,2)
    for I in 1:BLKSIZE:n
        for J in 1:BLKSIZE:n
            copy!(C_block, C, J+BLKSIZE, BLKSIZE * BLKSIZE)
            for K in 1:BLKSIZE:n
                copy!(A_block, A, K+BLKSIZE, BLKSIZE * BLKSIZE)
                copy!(A_block, B, I+BLKSIZE, BLKSIZE * BLKSIZE)
                kernel!(C_block, A_block, B_block, I, J, K, BLKSIZE)
                accum!(C, C_block)

            end
        end
    end
end

function kernel!(C_block, A_block, B_block, I, J, K, BLKSIZE)
    for i in I:I+BLKSIZE-1
        for j in J:J+BLKSIZE-1
            for k in K:K+BLKSIZE-1
                println(I, " ", J, " ", K)
                println(i, " ", j, " ", k)
                println("---------")
                C[i,j] += A[i,k]*B[k,j]
            end
        end
    end
end

    Ctile = pointer_to_array(convert(Ptr{R}, pointer(Cbuf)), sz)
    for jb = 1:tile_size:nB
        jlim = min(jb+tile_size-1,nB)
        jlen = jlim-jb+1
        for ib = 1:tile_size:mA
            ilim = min(ib+tile_size-1,mA)
            ilen = ilim-ib+1
            fill!(Ctile, z)
            for kb = 1:tile_size:nA
                klim = min(kb+tile_size-1,mB)
                klen = klim-kb+1
                Base.copy_transpose!(Atile, 1:klen, 1:ilen, tA, A, ib:ilim, kb:klim)
                copy!(Btile, 1:klen, 1:jlen, tB, B, kb:klim, jb:jlim)
                kernel
            end
            copy!(C, ib:ilim, jb:jlim, Ctile, 1:ilen, 1:jlen)
        end
    end

kernel()
for j=1:jlen
                    bcoff = (j-1)*tile_size
                    for i = 1:ilen
                        aoff = (i-1)*tile_size
                        s = z
                        for k = 1:klen
                            s += Atile[aoff+k] * Btile[bcoff+k]
                        end
                        Ctile[bcoff+i] += s
                    end
                end



for jb = 1:tile_size:nB
    for ib = 1:tile_size:mA
        for kb = 1:tile_size:nA
            update c
            kernel(n, )






    for i in I:BLKSIZE
        for j in J:BLKSIZE
            for k in J:BLKSIZE
                C[i,k] += A[i,k]*B[k,j]
            end
        end
    end
end
