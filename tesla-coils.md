
## Tesla Coils: high voltage resonant transformers

High voltage arcs.  Beautiful to witness!

Specs: ~4 kVA power input using an ASRG spark gap running at ~500 bps, made from a variable-speed angle grinder with a disk (fashioned from high density polyethylene cutting board) securing four flying brass electrodes.  The electrodes are brass as iron has high RF losses and can become too hot to use with polyethylene, which is a thermoplastic and melts under high temperatures. 

Tesla coils are air-cored resonant transformers, and are a type of LC circuit.  The L stands for inductor, and C for capacitor and when combined with a device that allows for rapid capacitor charging (the tank circuitpower supply) and discharging (the spark gap), The primary circuit capacitor is 120nF multi-mini style capacitor made from a few hundred surplus WIMA FKP1 pulse capacitors usually found in industrial lasers. 

The LC circuit is powered by 4 microwave oven transformers, primaries wired in parallel with secondaries in series (floating cores, and all submerged in mineral oil) for a power ouput of 10 kV at ~400mA.  Not pretty but a very inexpensive and robust power source.  In fact, one of the only components I have used that has not failed catastrophically at one point or another: even when a primary strike led to a vaporized grounding wire (connecting the secondary to the iron core) in one of the transformers, repairs were minimal.

![MOT stack]({{https://blbadger.github.io}}tesla_images/mot_stack.JPG)

One point of caution: the oil used here is plain mineral oil from the grocery store. In terms of flashover and arc resistance, industrial transformer oil is far superior.  But new transformer oil is quite expensive, and old oil can be hazardous. Be extra careful with any oil that has come out of a transformer, as it may contain polychlorinated biphenyls, aka PCBs, which are very toxic.  PCBs were routinely added to transformer oil before the 1970s, and any used transformer or oil purchased and suspected to originate in this era should be checked to be free of PCBs before use.

Power controlled with a variable autotransformer (for low power runs), up to 145V output with 120V input.  This variac allows a maximum current of around 3 1/2 kVA before saturating.

![variac]({{https://blbadger.github.io}}tesla_images/variac.JPG)

The topload is two aluminum dryer ducts forming a double stacked toroid, each stack 8" by 2.5'. Primary coil is ~8 turns of 0.25" refrigerator tubing, secondary coil is 1100 turns of AWG 22 magnet wire on an 8" inner diameter concrete  forming tube coated in polyurethane. The inner turns of the primary gets noticeably warm during operation, not suprising with an estimated 18.9 kA instantaneous current during capacitor discharge.

![tesla coil arcs]({{https://blbadger.github.io}}tesla_images/newtesla.jpg)

![tesla coil arcs]({{https://blbadger.github.io}}tesla_images/tesla_3.jpg)

![tesla coil arcs]({{https://blbadger.github.io}}tesla_images/tesla_4.jpg)

![tesla coil arcs]({{https://blbadger.github.io}}tesla_images/tesla_7.png)

### Low-res videos of the coil above

{% include youtube.html id='gwUA4ATNvRg' %}

![]()

{% include youtube.html id='FyRCdSQW1GY' %}


### Very large topload test

A rubber inner tube was inflated and covered in aluminum tape to make a very large toroidal topload (5' diameter).  This method is not recommended without the use of a forming substance (paper miche or plaster etc) covering the tube because the high voltage arcs will puncture the rubber even when protected via metal tape.  This is the last generation of the coil above, as the secondary base experienced severe flashovers resulting in vaporization of parts of the lower windings.  > 7' arcs!

![tesla coil arcs]({{https://blbadger.github.io}}tesla_images/large_tesla.gif)


### Early generation coil

This coil was based on a secondary coil wound on a 4" diameter PVC with 1300 turns of AWG 26 magnet wire. 3kVA input from the microwave oven transformer above but with an air core inductive current limiter, smaller dryer duct toroids, smaller primary capacitor (~ 50 nF) make from WIMA MKP-10 pulse capacitors (which are by design not as durable as FKP-1 but are cheaper).

![primary capacitor]({{https://blbadger.github.io}}tesla_images/wima_mkp10.JPG)

Note the bleed resistors at the top: I have not found these to be necessary for new capacitors (with no dielectric memory) used on tesla coils, but it is generally a good idea to include some form of current dissipation for large capacitors like these.

The primary and secondary coil setup on wood supports, with an angle grinder-based ASRG in the ground to the right and the capacitor bank underneath the primary.

![setup]({{https://blbadger.github.io}}tesla_images/old_tesla.JPG)

The jumper cables are attached to the strike rail protecting the primary coil and the secondary coil base, and lead to an 8' copper grounding rod.

![gen 1 tesla arcs 2]({{https://blbadger.github.io}}tesla_images/tesla_5.JPG)

![gen 1 tesla arcs]({{https://blbadger.github.io}}tesla_images/tesla_6.JPG)



