using Test
using ParameterizedConvexApproximators


n = 3
d = 1000
ratio1 = 0.7
ratio2 = 0.2


function test_split_data2()
    dataset = []
    for i in 1:d
        push!(dataset, rand(n))
    end
    dataset_train, dataset_test = split_data2(dataset, ratio1)
    @test length(dataset_train) == round(d * ratio1)
    @test length(dataset_test) == round(d * (1-ratio1))
end


function test_split_data3()
    dataset = []
    for i in 1:d
        push!(dataset, rand(n))
    end
    dataset_train, dataset_validate, dataset_test = split_data3(dataset, ratio1, ratio2)
    @test length(dataset_train) == round(d * ratio1)
    @test length(dataset_validate) == round(d * ratio2)
    @test length(dataset_test) == round(d * (1-(ratio1+ratio2)))
end


function main()
    test_split_data2()
    test_split_data3()
end


@testset "data_manipulation" begin
    main()
end
