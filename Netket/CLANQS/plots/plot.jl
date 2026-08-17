using CairoMakie, MakieHelpers, JSON, Statistics
function plot_kitaev_honeycomb_lattice!(ax)

    # Bond length = 1. Three NN vectors at 120° apart (from A to B sites)
    δz = (0.0, 1.0)                    # z-bond
    δx = (sqrt(3)/2, -0.5)             # x-bond
    δy = (-sqrt(3)/2, -0.5)            # y-bond

    L = 2
    # Lattice vectors connecting equivalent A sites
    a1 = δx .- δy   # = (sqrt(3), 0)
    a2 = δz .- δy   # = (sqrt(3)/2, 1.5)

    # fig = Figure(size = (650, 600))
    # ax = Axis(fig[1,1], aspect = DataAspect())
    # hidedecorations!(ax); hidespines!(ax)

    colors = Dict(:x => :crimson, :y => :seagreen, :z => :royalblue)

    sites_A = Point2f[]
    for m in -L:L, n in -L:L
        push!(sites_A, Point2f(m*a1[1] + n*a2[1], m*a1[2] + n*a2[2]))
    end

    for A in sites_A
        B = A .+ Point2f(δz...)

        # z-bond: within the same unit cell
        lines!(ax, [A[1], B[1]], [A[2], B[2]], color = colors[:z], linewidth = 5)

        # x-bond: B connects to A shifted by (a2 - a1)
        Ax = A .+ Point2f((a2 .- a1)...)
        lines!(ax, [B[1], Ax[1]], [B[2], Ax[2]], color = colors[:x], linewidth = 5)

        # y-bond: B connects to A shifted by a2
        Ay = A .+ Point2f(a2...)
        lines!(ax, [B[1], Ay[1]], [B[2], Ay[2]], color = colors[:y], linewidth = 5)
    end

    for A in sites_A
        B = A .+ Point2f(δz...)
        scatter!(ax, [A[1]], [A[2]], color = :black, markersize = 13)
        scatter!(ax, [B[1]], [B[2]], color = :white, strokecolor = :black,
                strokewidth = 1.5, markersize = 13)
    end

    xlims!(ax, -2, 2)
    ylims!(ax, -2.2, 1.8)

    elems = [LineElement(color = colors[k], linewidth = 5) for k in (:x, :y, :z)]
    # Legend(fig[1,2], elems, ["Jx", "Jy", "Jz"], "Bonds")

    # save("kitaev_honeycomb.png", fig)
    # fig
    ax
end
function plot_toric_code_lattice!(ax)

    L = 3  # lattice extends from 0 to L in both directions

    colors = Dict(:lattice => :gray70, :star => :royalblue, :plaquette => :crimson)

    # draw the underlying square lattice (thin gray edges)
    for i in 0:L, j in 0:L
        if i < L
            lines!(ax, [i, i+1], [j, j], color = colors[:lattice], linewidth = 2)
        end
        if j < L
            lines!(ax, [i, i], [j, j+1], color = colors[:lattice], linewidth = 2)
        end
    end

    # qubits live on edge midpoints
    qubits = Point2f[]
    for i in 0:L-1, j in 0:L
        push!(qubits, Point2f(i + 0.5, j))   # horizontal edges
    end
    for i in 0:L, j in 0:L-1
        push!(qubits, Point2f(i, j + 0.5))   # vertical edges
    end
    scatter!(ax, qubits, color = :white, strokecolor = :black,
             strokewidth = 1.5, markersize = 13)

    # highlight one star (vertex) operator: the 4 edges touching a chosen vertex
    v = Point2f(1, 1)
    star_edges = [
        (v, v .+ Point2f(1,0)),
        (v, v .+ Point2f(-1,0)),
        (v, v .+ Point2f(0,1)),
        (v, v .+ Point2f(0,-1)),
    ]
    for (p, q) in star_edges
        lines!(ax, [p[1], q[1]], [p[2], q[2]], color = colors[:star], linewidth = 5)
    end
    scatter!(ax, [v[1]], [v[2]], color = colors[:star], markersize = 10)

    # highlight one plaquette operator: the 4 edges around a chosen face
    p0 = Point2f(2, 1)  # bottom-left corner of the face
    plaq_corners = [p0, p0 .+ Point2f(1,0), p0 .+ Point2f(1,1), p0 .+ Point2f(0,1), p0]
    for k in 1:4
        p, q = plaq_corners[k], plaq_corners[k+1]
        lines!(ax, [p[1], q[1]], [p[2], q[2]], color = colors[:plaquette], linewidth = 5)
    end

    # redraw qubits on top of highlighted bonds so they stay visible
    scatter!(ax, qubits, color = :white, strokecolor = :black,
             strokewidth = 1.5, markersize = 12)

    xlims!(ax, -0.5, L+0.5)
    ylims!(ax, -0.5, L+0.5)

    ax
end
@views function rolling_avg(data, window_size)
    n = length(data)
    avg_data = zeros(n)
    for i in 1:n
        start_idx = max(1, i - window_size + 1)
        end_idx = i
        avg_data[i] = mean(data[start_idx:end_idx])
    end
    return avg_data
end

function get_iter_en(data; window_size=1)
    iters = Int.(data["Energy"]["iters"])
    en = float.(data["Energy"]["Mean"]["real"])
    if window_size > 1
        en_avg = rolling_avg(en, window_size)
    else
        en_avg = en
    end
    return iters, en_avg
end
##
cd(@__DIR__)

honeycomb_cliff = JSON.parsefile("../logs/kitaev_honeycomb_clifford.json")
honeycomb_original = JSON.parsefile("../logs/kitaev_honeycomb_original.json")
toric_cliff = JSON.parsefile("../logs/toric_code_perturbed_clifford.json")
toric_original = JSON.parsefile("../logs/toric_code_perturbed_original.json")
e_0_honey = honeycomb_original["exact_energy"]
e_0_toric = toric_original["exact_energy"]
##
with_theme(theme_SimpleTicks()) do 

    fig = Figure(size = (800, 600), fontsize = 20)
    axkwargs = (;xminorticksvisible = true, xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5), yminorticksvisible = true, )
    ax_tc = Axis(fig[1, 1], xlabel = "Iteration", ylabel = L"$|E - E_0|$", xticklabelsvisible = false, xlabelvisible = false; axkwargs...)
    axlog_tc = Axis(fig[2, 1], xlabel = "Iteration", ylabel = L"$\frac{|E - E_0|}{|E_0|}$", yscale = log10; axkwargs...)

    ax_hc = Axis(fig[1, 2], xlabel = "Iteration", xticklabelsvisible = false, xlabelvisible = false, yticklabelsvisible = false, ylabelvisible = false; axkwargs...)
    axlog_hc = Axis(fig[2, 2], xlabel = "Iteration", yscale = log10, yticklabelsvisible = false, ylabelvisible = false; axkwargs...)

    ax_hc_inset = insetAtPoint(fig, ax_hc, Point2f(3300, 13), (40,40), title = L"Kitaev Honeycomb $$", titlesize = 16)
    hidedecorations!(ax_hc_inset)
    ax_tc_inset = insetAtPoint(fig, ax_tc, Point2f(400, 13), (40,40), title = L"Toric Code $$", titlesize = 16)
    hidedecorations!(ax_tc_inset)

    plot_kitaev_honeycomb_lattice!(ax_hc_inset)
    plot_toric_code_lattice!(ax_tc_inset)

    let
        iters, en_avg = get_iter_en(toric_original, window_size=10)
        en_diff = en_avg .- e_0_toric
        lines!(ax_tc, iters, en_diff, label = L"Original$$", color = :grey)

        e_err = (en_avg .- e_0_toric) ./ abs(e_0_toric)
        lines!(axlog_tc, iters, e_err, label = L"Original$$", color = :grey)
    end
        
    let 
        iters, en_avg = get_iter_en(honeycomb_original, window_size=10)
        en_diff = en_avg .- e_0_honey
        lines!(ax_hc, iters, en_diff, label = L"Original$$", color = :grey)
        
        e_err = (en_avg .- e_0_honey) ./ abs(e_0_honey)
        lines!(axlog_hc, iters, e_err, label = L"Original$$", color = :grey)
    end

    let 
        iters, en_avg = get_iter_en(honeycomb_cliff, window_size=10)
        en_diff = en_avg .- e_0_honey
        lines!(ax_hc, iters, en_diff, label = L"optimized$$", color = :black, linewidth = 3)
        
        e_err = (en_avg .- e_0_honey) ./ abs(e_0_honey)
        lines!(axlog_hc, iters, e_err, label = L"optimized$$", color = :black, linewidth = 3)
    end
        let
        iters, en_avg = get_iter_en(toric_cliff, window_size=10)
        en_diff = en_avg .- e_0_toric
        lines!(ax_tc, iters, en_diff, label = L"optimized$$", color = :black, linewidth = 3)
        
        e_err = (en_avg .- e_0_toric) ./ abs(e_0_toric)
        lines!(axlog_tc, iters, e_err, label = L"optimized$$", color = :black, linewidth = 3)
    end

    axislegend(ax_tc, position = :rt, framevisible = false,merge=true)
    # axislegend(axlog, position = :rt, framevisible = false)

    linkxaxes!(ax_tc, axlog_tc)
    linkxaxes!(ax_hc, axlog_hc)
    linkyaxes!(axlog_tc, axlog_hc)
    linkyaxes!(ax_tc, ax_hc)
    save("kitaev_toric_energy_convergence.pdf", fig)
    fig
    
end

##
with_theme(theme_SimpleTicks()) do 

    fig = Figure(size = (1100, 400), fontsize = 20)
    axkwargs = (;xminorticksvisible = true, xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5), yminorticksvisible = true, )
    axlog_tc = Axis(fig[1, 1], xlabel = "Iteration", ylabel = L"$\frac{|E - E_0|}{|E_0|}$", yscale = log10; axkwargs...)

    axlog_hc = Axis(fig[1, 2], xlabel = "Iteration", yscale = log10,
    # yticklabelsvisible = false, ylabelvisible = false
    ; axkwargs...)

    ax_hc_inset = insetAtPoint(fig, axlog_hc, Point2f(3500, -0.07), (50,50), title = L"Honeycomb $$", titlesize = 16)
    hidedecorations!(ax_hc_inset)
    ax_tc_inset = insetAtPoint(fig, axlog_tc, Point2f(3500, -1), (50,50), title = L"Toric Code $$", titlesize = 16)
    hidedecorations!(ax_tc_inset)

    plot_kitaev_honeycomb_lattice!(ax_hc_inset)
    plot_toric_code_lattice!(ax_tc_inset)

    let
        iters, en_avg = get_iter_en(toric_original, window_size=10)
        en_diff = en_avg .- e_0_toric

        e_err = abs.((en_avg .- e_0_toric) ./ e_0_toric)
        lines!(axlog_tc, iters, e_err, label = L"Original$$", color = :grey)
    end
        
    let 
        iters, en_avg = get_iter_en(honeycomb_original, window_size=10)
        en_diff = en_avg .- e_0_honey
        e_err = abs.((en_avg .- e_0_honey) ./ e_0_honey)
        lines!(axlog_hc, iters, e_err, label = L"Original$$", color = :grey)
    end

    let 
        iters, en_avg = get_iter_en(honeycomb_cliff, window_size=10)
        en_diff = en_avg .- e_0_honey
        e_err = abs.((en_avg .- e_0_honey) ./ e_0_honey)
        lines!(axlog_hc, iters, e_err, label = L"optimized$$", color = :black, linewidth = 3)
    end
    let
        iters, en_avg = get_iter_en(toric_cliff, window_size=10)
        en_diff = en_avg .- e_0_toric
        e_err = abs.((en_avg .- e_0_toric) ./ e_0_toric)
        lines!(axlog_tc, iters, e_err, label = L"optimized$$", color = :black, linewidth = 3)
    end

    axislegend(axlog_tc, position = :ct, framevisible = false,merge=true)
    # axislegend(axlog, position = :rt, framevisible = false)

    # linkxaxes!(ax_tc, axlog_tc)
    # linkxaxes!(ax_hc, axlog_hc)
    # linkyaxes!(axlog_tc, axlog_hc)
    # linkyaxes!(ax_tc, ax_hc)
    save("kitaev_toric_energy_convergence.pdf", fig)
    fig
    
end