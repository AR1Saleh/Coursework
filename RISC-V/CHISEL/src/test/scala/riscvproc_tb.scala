import chisel3._
import chiseltest._
import chiseltest.simulator.WriteVcdAnnotation
import org.scalatest.flatspec.AnyFlatSpec

class CoreModulesTester extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "RiscV Core Reset Sequence"

  it should "generate VCD file" in {
    test(new RiscV).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step()
      dut.reset.poke(false.B)
      // Add stimulus here to generate signal activity
      dut.clock.step(20)
    }
  }
}
